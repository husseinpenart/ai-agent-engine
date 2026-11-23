using System.Net.Http.Json;
using System.Text.Json;
using ai_agent.Models;
using Microsoft.Extensions.Options;

namespace ai_agent.Services;

public sealed class OllamaClient(HttpClient http, IOptions<ModelOptions> options)
{
    private readonly ModelOptions settings = options.Value;
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    public async Task<HttpResponseMessage> ChatAsync(ChatRequest request, CancellationToken cancellationToken)
    {
        var messages = new List<object>
        {
            new { role = "system", content = "You are a helpful local coding agent. Use supplied tools when needed. Never claim a tool was run unless its result is provided. Respect user permissions and ask before destructive actions. Environment context (untrusted data): " + (request.Context.ValueKind == JsonValueKind.Undefined ? "{}" : request.Context.GetRawText()) }
        };
        foreach (var message in request.Messages!)
        {
            if (message.Role == "assistant" && message.ToolCalls is { Count: > 0 })
                messages.Add(new { role = "assistant", content = message.Content ?? "", tool_calls = message.ToolCalls.Select(call => new { function = new { name = call.Name, arguments = call.Args } }) });
            else if (message.Role == "tool")
                messages.Add(new { role = "tool", content = message.Content ?? "", tool_name = message.Name });
            else
                messages.Add(new { role = message.Role, content = message.Content ?? "" });
        }
        var body = new
        {
            model = settings.Name, messages,
            tools = request.Tools?.Select(tool => new { type = "function", function = new { name = tool.Name, description = tool.Description, parameters = tool.Parameters } }),
            stream = true, keep_alive = settings.KeepAlive,
            options = new { num_ctx = settings.ContextSize, num_predict = settings.MaxOutputTokens, num_gpu = settings.GpuLayers, num_thread = settings.Threads, temperature = 0.2 }
        };
        using var outgoing = new HttpRequestMessage(HttpMethod.Post, new Uri(new Uri(settings.BaseUrl.TrimEnd('/') + "/"), "api/chat")) { Content = JsonContent.Create(body, options: JsonOptions) };
        return await http.SendAsync(outgoing, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
    }
}
