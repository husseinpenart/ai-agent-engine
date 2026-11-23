using System.Text.Json;
using System.Text;
using ai_agent.Models;
using ai_agent.Services;
using Microsoft.AspNetCore.Mvc;

namespace ai_agent.Controllers;

[ApiController]
public sealed class AgentController(LocalModelClient client, ModelProcess runtime, InferenceGate gate, ILogger<AgentController> logger) : ControllerBase
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    [HttpGet("/api/health")]
    public IActionResult Health() => Ok(new { status = "ok" });

    [HttpPost("/api/agent/chat")]
    [RequestSizeLimit(2_000_000)]
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
        if (!await gate.Semaphore.WaitAsync(0, cancellationToken))
        {
            Response.StatusCode = 429;
            await Response.WriteAsJsonAsync(new { error = "The model is already processing a request." }, cancellationToken);
            return;
        }
        try
        {
            using var upstream = await client.ChatAsync(request, cancellationToken);
            if (!upstream.IsSuccessStatusCode)
            {
                Response.StatusCode = 502;
                await Response.WriteAsJsonAsync(new { error = "Model server returned " + (int)upstream.StatusCode + ": " + await upstream.Content.ReadAsStringAsync(cancellationToken) }, cancellationToken);
                return;
            }
            Response.ContentType = "text/event-stream";
            Response.Headers.CacheControl = "no-cache";
            await using var stream = await upstream.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);
            var complete = false;
            var toolCalls = new SortedDictionary<int, (StringBuilder Id, StringBuilder Name, StringBuilder Arguments)>();
            while (await reader.ReadLineAsync(cancellationToken) is { } line)
            {
                if (!line.StartsWith("data:", StringComparison.Ordinal)) continue;
                var data = line[5..].Trim();
                if (data == "[DONE]") { complete = true; break; }
                using var document = JsonDocument.Parse(data);
                var root = document.RootElement;
                if (root.TryGetProperty("error", out var error)) { await Send(new { type = "error", message = error.ToString() }, cancellationToken); return; }
                if (root.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0 && choices[0].TryGetProperty("delta", out var delta))
                {
                    if (delta.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String && content.GetString() is { Length: > 0 } text) await Send(new { type = "text", delta = text }, cancellationToken);
                    if (delta.TryGetProperty("reasoning_content", out var thinking) && thinking.ValueKind == JsonValueKind.String && thinking.GetString() is { Length: > 0 } thought) await Send(new { type = "thinking", delta = thought }, cancellationToken);
                    if (delta.TryGetProperty("tool_calls", out var calls) && calls.ValueKind == JsonValueKind.Array)
                        foreach (var call in calls.EnumerateArray())
                        {
                            var index = call.GetProperty("index").GetInt32();
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
            if (!complete) { await Send(new { type = "error", message = "Model stream ended unexpectedly." }, cancellationToken); return; }
            foreach (var call in toolCalls.Values)
            {
                using var arguments = JsonDocument.Parse(call.Arguments.Length == 0 ? "{}" : call.Arguments.ToString());
                await Send(new { type = "tool_call", id = call.Id.Length == 0 ? Guid.NewGuid().ToString("N") : call.Id.ToString(), name = call.Name.ToString(), args = arguments.RootElement }, cancellationToken);
            }
            await Send(new { type = "done" }, cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { }
        catch (Exception exception)
        {
            logger.LogError(exception, "Local inference failed");
            if (!Response.HasStarted) { Response.StatusCode = 503; await Response.WriteAsJsonAsync(new { error = "Local model could not start: " + exception.Message }, cancellationToken); }
            else await Send(new { type = "error", message = "Local inference failed. Check backend logs." }, cancellationToken);
        }
        finally { runtime.ScheduleStop(); gate.Semaphore.Release(); }
    }

    private async Task Send(object value, CancellationToken cancellationToken)
    {
        await Response.WriteAsync("data: " + JsonSerializer.Serialize(value, JsonOptions) + "\n\n", cancellationToken);
        await Response.Body.FlushAsync(cancellationToken);
    }
}
