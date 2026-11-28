using System.Text.Json;
using System.Text.RegularExpressions;
using ai_agent.Models;
using ai_agent.Services;
using Microsoft.AspNetCore.Mvc;

namespace ai_agent.Controllers;

[ApiController]
public sealed partial class AgentController(
    LocalModelClient client,
    VisionClient vision,
    ModelProcess runtime,
    InferenceGate gate,
    ILogger<AgentController> logger) : ControllerBase
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);
    private readonly SemaphoreSlim _responseLock = new(1, 1);

    [GeneratedRegex(@"\[(?:Executed\s+)?Tool calls:\s*\[.*?\]\]", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex ToolCallsBracketRegex();

    [HttpGet("/api/health")]
    public IActionResult Health() => Ok(new { status = "ok" });

    [HttpPost("/api/agent/chat")]
    [RequestSizeLimit(15_000_000)]
    [ProducesResponseType(typeof(string), StatusCodes.Status200OK, "text/event-stream")]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status503ServiceUnavailable)]
    public async Task Chat([FromBody] ChatRequest request, CancellationToken cancellationToken)
    {
        if (request.Messages is not { Count: > 0 and <= 200 } ||
            request.Messages.Any(m => m.Role is not ("user" or "assistant" or "tool")) ||
            request.Tools?.Count > 40)
        {
            Response.StatusCode = 400;
            await Response.WriteAsJsonAsync(new { error = "Invalid messages or tools." }, cancellationToken);
            return;
        }

        if (request.Images is { Count: > 0 } images &&
            (images.Count > 4 ||
             images.Any(image => image is null || !VisionClient.IsValidImage(image)) ||
             images.Sum(image => (long)(image.DataUrl?.Length ?? 0)) > 12_000_000 ||
             images.Select(image => image.Id).Distinct().Count() != images.Count ||
             request.Messages.All(message => message.Role != "user")))
        {
            Response.StatusCode = 400;
            await Response.WriteAsJsonAsync(
                new { error = "Invalid image data. Use up to four PNG, JPEG, or WebP data URLs." },
                cancellationToken);
            return;
        }

        Response.ContentType = "text/event-stream";
        Response.Headers.CacheControl = "no-cache";

        logger.LogInformation("Chat request started for session {SessionId} with {MessageCount} messages and {ImageCount} images",
            request.SessionId, request.Messages.Count, request.Images?.Count ?? 0);

        await Send(new { type = "status", phase = "queued" }, cancellationToken);

        using var heartbeatCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var heartbeat = KeepAlive(heartbeatCancellation.Token);

        var enteredGate = false;

        try
        {
            await gate.Semaphore.WaitAsync(cancellationToken);
            enteredGate = true;

            var reports = new List<string>();
            if (request.Images is { Count: > 0 })
            {
                logger.LogInformation("Running vision analysis for session {SessionId}", request.SessionId);
                await Send(new { type = "status", phase = "vision" }, cancellationToken);
                try
                {
                    var userText = request.Messages.Last(message => message.Role == "user").Content ?? "";
                    var report = await vision.AnalyzeAsync(request.Images, userText, cancellationToken);
                    reports.Add(report);
                    for (var index = 0; index < request.Images.Count; index++)
                    {
                        var image = request.Images[index];
                        var imageReport = index == 0
                            ? report
                            : $"Included in the joint visual analysis for {request.Images[0].Name} (image {index + 1}: {image.Name}).";
                        await Send(new { type = "vision_report", id = image.Id, report = imageReport }, cancellationToken);
                    }
                }
                finally
                {
                    runtime.UnloadVision();
                }
            }

            await Send(new { type = "status", phase = "coding" }, cancellationToken);
            await Send(new { type = "status", phase = "inference" }, cancellationToken);

            logger.LogInformation("Running coder inference for session {SessionId}", request.SessionId);
            var rawReply = await client.GenerateReplyAsync(request, reports, cancellationToken);

            var calls = ToolCallParser.Extract(rawReply, LocalModelClient.ResolveTools(request));
            var cleanText = ToolCallsBracketRegex().Replace(ToolCallParser.Strip(rawReply), string.Empty).Trim();

            logger.LogInformation("Coder inference finished for session {SessionId}: {ToolCount} tool calls, {TextLength} chars",
                request.SessionId, calls.Count, cleanText.Length);

            if (calls.Count > 0)
            {
                foreach (var call in calls)
                {
                    await Send(new { type = "tool_call", id = call.Id, name = call.Name, args = call.Args }, cancellationToken);
                }
            }
            else if (!string.IsNullOrEmpty(cleanText))
            {
                await Send(new { type = "text", delta = cleanText }, cancellationToken);
            }
            else if (!string.IsNullOrWhiteSpace(rawReply))
            {
                await Send(new { type = "text", delta = "خروجی ابزار در میانه تولید ناقص شد؛ لطفاً یک بار دیگر درخواست را ارسال کن تا تغییر در بخش‌های کوتاه‌تر اعمال شود." }, cancellationToken);
            }
            else
            {
                await Send(new { type = "text", delta = "مدل خروجی متنی یا ابزار معتبری برنگرداند. لطفاً درخواست را دوباره ارسال کنید." }, cancellationToken);
            }

            await Send(new { type = "done", finishReason = "stop" }, cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            logger.LogInformation("Chat request cancelled for session {SessionId}", request.SessionId);
        }
        catch (Exception exception)
        {
            logger.LogError(exception, "Local inference failed for session {SessionId}", request.SessionId);
            if (!Response.HasStarted)
            {
                Response.StatusCode = 503;
                await Response.WriteAsJsonAsync(new { error = "Local model could not start: " + exception.Message }, cancellationToken);
            }
            else if (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    await Send(new { type = "error", message = "Local inference failed: " + exception.Message }, CancellationToken.None);
                }
                catch
                {
                }
            }
        }
        finally
        {
            try
            {
                heartbeatCancellation.Cancel();
                await heartbeat;
            }
            catch
            {
            }

            if (enteredGate)
            {
                try
                {
                    runtime.ScheduleStop();
                }
                catch (Exception exception)
                {
                    logger.LogError(exception, "Failed to schedule model unload after request");
                }
                finally
                {
                    gate.Semaphore.Release();
                }
            }
        }
    }

    private async Task Send(object value, CancellationToken cancellationToken)
    {
        await WriteEvent("data: " + JsonSerializer.Serialize(value, JsonOptions) + "\n\n", cancellationToken);
    }

    private async Task WriteEvent(string data, CancellationToken cancellationToken)
    {
        await _responseLock.WaitAsync(cancellationToken);
        try
        {
            await Response.WriteAsync(data, cancellationToken);
            await Response.Body.FlushAsync(cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
        catch (IOException exception)
        {
            logger.LogDebug(exception, "Unable to write SSE event to response stream");
        }
        finally
        {
            _responseLock.Release();
        }
    }

    private async Task KeepAlive(CancellationToken cancellationToken)
    {
        try
        {
            using var timer = new PeriodicTimer(TimeSpan.FromSeconds(15));
            while (await timer.WaitForNextTickAsync(cancellationToken))
            {
                await WriteEvent(": keep-alive\n\n", cancellationToken);
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (IOException exception)
        {
            logger.LogDebug(exception, "Unable to send stream keep-alive");
        }
    }
}