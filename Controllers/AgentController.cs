using System.Text.Json;
using ai_agent.Models;
using ai_agent.Services;
using Microsoft.AspNetCore.Mvc;

namespace ai_agent.Controllers;

[ApiController]
public sealed class AgentController(OllamaClient client, InferenceGate gate, ILogger<AgentController> logger) : ControllerBase
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    [HttpGet("/api/health")]
    public IActionResult Health() => Ok(new { status = "ok" });

    [HttpPost("/api/agent/chat")]
    [RequestSizeLimit(2_000_000)]
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
            while (await reader.ReadLineAsync(cancellationToken) is { } line)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                using var document = JsonDocument.Parse(line);
                var root = document.RootElement;
                if (root.TryGetProperty("error", out var error)) { await Send(new { type = "error", message = error.GetString() }, cancellationToken); return; }
                if (root.TryGetProperty("message", out var message))
                {
                    if (message.TryGetProperty("content", out var content) && content.GetString() is { Length: > 0 } text) await Send(new { type = "text", delta = text }, cancellationToken);
                    if (message.TryGetProperty("thinking", out var thinking) && thinking.GetString() is { Length: > 0 } thought) await Send(new { type = "thinking", delta = thought }, cancellationToken);
                    if (message.TryGetProperty("tool_calls", out var calls) && calls.ValueKind == JsonValueKind.Array)
                        foreach (var call in calls.EnumerateArray())
                        {
                            var function = call.GetProperty("function");
                            await Send(new { type = "tool_call", id = Guid.NewGuid().ToString("N"), name = function.GetProperty("name").GetString(), args = function.GetProperty("arguments") }, cancellationToken);
                        }
                }
                if (root.TryGetProperty("done", out var done) && done.ValueKind == JsonValueKind.True) { complete = true; break; }
            }
            if (!complete) { await Send(new { type = "error", message = "Model stream ended unexpectedly." }, cancellationToken); return; }
            await Send(new { type = "done" }, cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { }
        catch (Exception exception)
        {
            logger.LogError(exception, "Local inference failed");
            if (!Response.HasStarted) { Response.StatusCode = 503; await Response.WriteAsJsonAsync(new { error = "Cannot reach the local model server. Check Ollama and the configured model." }, cancellationToken); }
            else await Send(new { type = "error", message = "Local inference failed. Check backend logs." }, cancellationToken);
        }
        finally { gate.Semaphore.Release(); }
    }

    private async Task Send(object value, CancellationToken cancellationToken)
    {
        await Response.WriteAsync("data: " + JsonSerializer.Serialize(value, JsonOptions) + "\n\n", cancellationToken);
        await Response.Body.FlushAsync(cancellationToken);
    }
}
