using System.Text.Json;
using System.Text;
using ai_agent.Models;
using ai_agent.Services;
using Microsoft.AspNetCore.Mvc;

namespace ai_agent.Controllers;

[ApiController]
public sealed class AgentController(LocalModelClient client, VisionClient vision, ModelProcess runtime, InferenceGate gate, ILogger<AgentController> logger) : ControllerBase
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);
    private readonly SemaphoreSlim responseLock = new(1, 1);

    [HttpGet("/api/health")]
    public IActionResult Health() => Ok(new { status = "ok" });

    [HttpPost("/api/agent/chat")]
    [RequestSizeLimit(10_000_000)]
    [ProducesResponseType(typeof(string), StatusCodes.Status200OK, "text/event-stream")]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status429TooManyRequests)]
    [ProducesResponseType(StatusCodes.Status502BadGateway)]
    [ProducesResponseType(StatusCodes.Status503ServiceUnavailable)]
    public async Task Chat([FromBody] ChatRequest request, CancellationToken cancellationToken)
    {
        if (request.Messages is not { Count: > 0 and <= 200 } || request.Messages.Any(m => m.Role is not ("user" or "assistant" or "tool")) || request.Tools?.Count > 40)
        {
            Response.StatusCode = 400;
            await Response.WriteAsJsonAsync(new { error = "Invalid messages or tools." }, cancellationToken);
            return;
        }
        if (request.Images is { Count: > 0 } images && (images.Count > 4 || images.Any(image => image is null || !VisionClient.IsValidImage(image)) || images.Sum(image => (long)(image.DataUrl?.Length ?? 0)) > 6_700_000 || images.Select(image => image.Id).Distinct().Count() != images.Count || request.Messages.All(message => message.Role != "user")))
        {
            Response.StatusCode = 400;
            await Response.WriteAsJsonAsync(new { error = "Invalid image data. Use up to four PNG, JPEG, or WebP data URLs, with at most 5 MB total." }, cancellationToken);
            return;
        }
        Response.ContentType = "text/event-stream";
        Response.Headers.CacheControl = "no-cache";
        await Send(new { type = "status", phase = "queued" }, cancellationToken);
        using var heartbeatCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var heartbeat = KeepAlive(heartbeatCancellation.Token);
        var enteredGate = false;
        var finished = false;
        try
        {
            await gate.Semaphore.WaitAsync(cancellationToken);
            enteredGate = true;
            var reports = new List<string>();
            if (request.Images is { Count: > 0 })
            {
                await Send(new { type = "status", phase = "vision" }, cancellationToken);
                try
                {
                    var userText = request.Messages.Last(message => message.Role == "user").Content ?? "";
                    foreach (var image in request.Images)
                    {
                        var report = await vision.AnalyzeAsync(image, userText, cancellationToken);
                        reports.Add($"Image: {image.Name}\n{report}");
                        await Send(new { type = "vision_report", id = image.Id, report }, cancellationToken);
                    }
                }
                finally { runtime.UnloadVision(); }
            }
            await Send(new { type = "status", phase = "coding" }, cancellationToken);
            using var upstream = await client.ChatAsync(request, reports, cancellationToken);
            if (!upstream.IsSuccessStatusCode)
            {
                var detail = "Model server returned " + (int)upstream.StatusCode + ": " + await upstream.Content.ReadAsStringAsync(cancellationToken);
                if (Response.HasStarted) await Send(new { type = "error", message = detail }, cancellationToken);
                else { Response.StatusCode = 502; await Response.WriteAsJsonAsync(new { error = detail }, cancellationToken); }
                return;
            }
            if (!Response.HasStarted) { Response.ContentType = "text/event-stream"; Response.Headers.CacheControl = "no-cache"; }
            await Send(new { type = "status", phase = "inference" }, cancellationToken);
            await using var stream = await upstream.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);
            var complete = false;
            var emittedText = false;
            string? finishReason = null;
            var toolCalls = new SortedDictionary<int, (StringBuilder Id, StringBuilder Name, StringBuilder Arguments)>();
            var highestToolIndex = -1;
            var outOfOrder = false;
            try
            {
            while (await reader.ReadLineAsync(cancellationToken) is { } line)
            {
                if (!line.StartsWith("data:", StringComparison.Ordinal)) continue;
                var data = line[5..].Trim();
                if (data == "[DONE]") { complete = true; break; }
                using var document = JsonDocument.Parse(data);
                var root = document.RootElement;
                if (root.TryGetProperty("error", out var error)) { await Send(new { type = "error", message = error.ToString() }, cancellationToken); return; }
                if (root.TryGetProperty("choices", out var finishChoices) && finishChoices.GetArrayLength() > 0 && finishChoices[0].TryGetProperty("finish_reason", out var finish) && finish.ValueKind == JsonValueKind.String) finishReason = finish.GetString();
                if (root.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0 && choices[0].TryGetProperty("delta", out var delta))
                {
                    if (delta.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String && content.GetString() is { Length: > 0 } text) { emittedText = true; await Send(new { type = "text", delta = text }, cancellationToken); }
                    if (delta.TryGetProperty("reasoning_content", out var thinking) && thinking.ValueKind == JsonValueKind.String && thinking.GetString() is { Length: > 0 } thought) await Send(new { type = "thinking", delta = thought }, cancellationToken);
                    if (delta.TryGetProperty("tool_calls", out var calls) && calls.ValueKind == JsonValueKind.Array)
                        foreach (var call in calls.EnumerateArray())
                        {
                            var index = call.GetProperty("index").GetInt32();
                            if (index < highestToolIndex) outOfOrder = true;
                            highestToolIndex = Math.Max(highestToolIndex, index);
                            if (!toolCalls.TryGetValue(index, out var pending)) { pending = (new(), new(), new()); toolCalls.Add(index, pending); }
                            if (call.TryGetProperty("id", out var id) && id.ValueKind == JsonValueKind.String) pending.Id.Append(id.GetString());
                            if (call.TryGetProperty("function", out var function))
                            {
                                if (function.TryGetProperty("name", out var name) && name.ValueKind == JsonValueKind.String) pending.Name.Append(name.GetString());
                                if (function.TryGetProperty("arguments", out var args) && args.ValueKind == JsonValueKind.String) pending.Arguments.Append(args.GetString());
                            }
                        }
                }
            }
            }
            catch (Exception exception) when (!cancellationToken.IsCancellationRequested && exception is IOException or HttpRequestException)
            {
                logger.LogWarning(exception, "Model stream connection failed before completion");
            }
            if (!complete)
            {
                var processDetail = runtime.ExitedProcessDetail();
                logger.LogWarning("Model stream ended before its completion marker; partial tool calls will not execute. {ProcessDetail}", processDetail);
                if (!emittedText && toolCalls.Count == 0) { await Send(new { type = "error", message = processDetail ?? "Model stream ended before producing a response." }, cancellationToken); return; }
                finishReason = "interrupted";
            }
            if (finishReason == "length") logger.LogWarning("Model output reached its token or context limit; continuation or safe tool retry is required");
            var pendingCalls = toolCalls.Select(item => new PendingToolCall(item.Key, item.Value.Id.Length == 0 ? Guid.NewGuid().ToString("N") : item.Value.Id.ToString(), item.Value.Name.ToString(), item.Value.Arguments.ToString()));
            var parsedCalls = ToolCallValidator.ParseBatch(pendingCalls, finishReason is "length" or "interrupted", outOfOrder);
            var missingName = parsedCalls.Any(call => string.IsNullOrEmpty(call.Name));
            var incomplete = parsedCalls.Any(call => call.Error is not null);
            const string retry = "Check the results of earlier calls before retrying. Retry the failed call with complete JSON and at most 6000 content characters. Use write_file only to start a file and append_file for subsequent chunks; do not overwrite earlier successful work.";
            if (incomplete) logger.LogWarning("Rejected incomplete or invalid model tool calls. Missing name: {MissingName}; reasons: {Reasons}", missingName, string.Join("; ", parsedCalls.Where(call => call.Error is not null).Select(call => call.Error)));
            foreach (var call in parsedCalls.Where(call => !string.IsNullOrEmpty(call.Name)))
                await Send(new { type = "tool_call", id = call.Id, name = call.Name, args = call.Arguments, error = call.Error is not null ? call.Error + " " + retry : null }, cancellationToken);
            if (missingName) await Send(new { type = "tool_retry", message = "A tool call was incomplete and had no name. " + retry }, cancellationToken);
            await Send(new { type = "done", finishReason }, cancellationToken);
            finished = complete;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { }
        catch (Exception exception)
        {
            logger.LogError(exception, "Local inference failed");
            if (!Response.HasStarted) { Response.StatusCode = 503; await Response.WriteAsJsonAsync(new { error = "Local model could not start: " + exception.Message }, cancellationToken); }
            else await Send(new { type = "error", message = "Local inference failed: " + exception.Message }, cancellationToken);
        }
        finally
        {
            heartbeatCancellation.Cancel();
            await heartbeat;
            if (enteredGate)
            {
                try { if (finished && !cancellationToken.IsCancellationRequested) runtime.ScheduleStop(); else runtime.Stop(); }
                catch (Exception exception) { logger.LogError(exception, "Failed to stop the model after a request"); }
                finally { gate.Semaphore.Release(); }
            }
        }
    }

    private async Task Send(object value, CancellationToken cancellationToken)
    {
        await WriteEvent("data: " + JsonSerializer.Serialize(value, JsonOptions) + "\n\n", cancellationToken);
    }

    private async Task WriteEvent(string data, CancellationToken cancellationToken)
    {
        await responseLock.WaitAsync(cancellationToken);
        try { await Response.WriteAsync(data, cancellationToken); await Response.Body.FlushAsync(cancellationToken); }
        finally { responseLock.Release(); }
    }

    private async Task KeepAlive(CancellationToken cancellationToken)
    {
        try
        {
            using var timer = new PeriodicTimer(TimeSpan.FromSeconds(15));
            while (await timer.WaitForNextTickAsync(cancellationToken)) await WriteEvent(": keep-alive\n\n", cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { }
        catch (IOException exception) { logger.LogDebug(exception, "Unable to send stream keep-alive"); }
    }
}
