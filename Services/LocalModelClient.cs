using System.Text;
using System.Text.Json;
using ai_agent.Models;
using Microsoft.Extensions.Options;

namespace ai_agent.Services;

public sealed class LocalModelClient(
    ModelProcess runtime,
    ConversationContext contextManager,
    IOptions<ModelOptions> options,
    ILogger<LocalModelClient> logger)
{
    private static readonly JsonSerializerOptions Web = new(JsonSerializerDefaults.Web);
    private static readonly string[] SchemaKeys = ["parameters", "input_schema", "inputSchema"];
    private readonly ModelOptions _settings = options.Value;

    private const string Instructions = """
You are an autonomous coding agent working inside the user's connected project workspace.
Learn the project's language, framework and conventions from its files before changing anything.

Rules:
1. Explore minimally: list or search once or twice, then read_file the files you need.
2. ALWAYS read a file before editing it. For edit_file, old_str must be copied character-for-character from the most recent read_file output (same indentation) and must appear exactly once; include a few surrounding lines to make it unique. Prefer small targeted edits. Use write_file only for new files or complete rewrites.
3. If edit_file or write_file returns an error, read the file again and retry with corrected text. Never guess file contents.
4. Do not repeat a tool call whose result is already in the conversation unless the file changed since.
5. Never invent tool results, and never say something was changed unless a tool result confirmed it.
6. When the task is finished, reply with a short summary in the user's language (Persian if the user writes Persian). Do not include tool-call syntax in that final reply.
""";

    private const string CallFormat = """
If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- Call ONE function per reply, then wait for its result
- Parameter values are raw text: do NOT escape, quote or JSON-encode code or file contents
- You may give brief reasoning before the function call, but NOT after
</IMPORTANT>
""";

    // Used only when the frontend does not send its own `tools` list.
    private const string FallbackTools = """
[
 {"name":"list_files","description":"List files in the workspace (recursive, ignores build and dependency folders). Optionally filter by a substring of the path.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Sub-directory relative to the workspace root. Default: root."},"pattern":{"type":"string","description":"Substring the file path must contain."}}}},
 {"name":"search_files","description":"Find files whose relative path contains the query.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Substring to look for in file paths."}},"required":["query"]}},
 {"name":"read_file","description":"Read a text file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path relative to the workspace root."}},"required":["path"]}},
 {"name":"write_file","description":"Create a new file or completely overwrite an existing one.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path relative to the workspace root."},"content":{"type":"string","description":"Full file content."}},"required":["path","content"]}},
 {"name":"edit_file","description":"Replace one exact block of text in an existing file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path relative to the workspace root."},"old_str":{"type":"string","description":"Exact existing text, unique in the file."},"new_str":{"type":"string","description":"Replacement text."}},"required":["path","old_str","new_str"]}},
 {"name":"append_file","description":"Append text to the end of an existing file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path relative to the workspace root."},"content":{"type":"string","description":"Text to append."}},"required":["path","content"]}}
]
""";

    public static JsonElement ResolveTools(ChatRequest request)
    {
        if (request.Tools is { Count: > 0 })
            return JsonSerializer.SerializeToElement(request.Tools, Web);

        using var doc = JsonDocument.Parse(FallbackTools);
        return doc.RootElement.Clone();
    }

    public async Task<string> GenerateReplyAsync(
        ChatRequest request,
        IReadOnlyList<string> visionReports,
        CancellationToken cancellationToken)
    {
        var tools = ResolveTools(request);

        var contextJson = request.Context.ValueKind == JsonValueKind.Undefined ? "{}" : request.Context.GetRawText();
        if (contextJson.Length > 3000) contextJson = contextJson[..3000] + "…";

        var (recent, summary) = contextManager.Prepare(request.Messages!);

        var system = new StringBuilder(Instructions);
        system.Append("\n\nWorkspace context (JSON): ").Append(contextJson);
        if (!string.IsNullOrWhiteSpace(summary))
            system.Append("\n\nEarlier conversation (summary):\n").Append(summary);
        system.Append("\n\n").Append(RenderTools(tools));
        var systemText = system.ToString();

        // ---- build turns -------------------------------------------------
        var toolLimit = Math.Clamp(_settings.ToolResultMaxChars, 2000, 60000);
        var turns = new List<Turn>();
        var lastUser = recent.FindLastIndex(m => m.Role == "user");

        for (var i = 0; i < recent.Count; i++)
        {
            var m = recent[i];
            var content = m.Content ?? "";

            if (m.Role == "tool")
            {
                var body = "<tool_response>\n" + Clip(content, toolLimit) + "\n</tool_response>";
                if (turns.Count > 0 && turns[^1].IsToolResult)
                    turns[^1].Text += "\n" + body;
                else
                    turns.Add(new Turn("user", body, true));
            }
            else if (m.Role == "assistant")
            {
                var text = content.Trim();
                if (m.ToolCalls is { Count: > 0 })
                {
                    foreach (var call in m.ToolCalls)
                        text += (text.Length > 0 ? "\n\n" : "") + RenderCall(call.Name, call.Args);
                }
                turns.Add(new Turn("assistant", text, false));
            }
            else
            {
                if (i == lastUser && visionReports.Count > 0)
                    content += "\n\n[Visual analysis of the attached images]\n" + string.Join("\n\n", visionReports);
                turns.Add(new Turn("user", content, false));
            }
        }

        // ---- fit into the context window --------------------------------
        var budget = Math.Max(2048, _settings.ContextSize - Math.Min(_settings.MaxOutputTokens, 4096) - 300);
        int Total() => Est(systemText) + turns.Sum(t => Est(t.Text) + 8);

        for (var guard = 0; guard < 300 && Total() > budget; guard++)
        {
            var big = turns.FindIndex(t => t.IsToolResult && t.Text.Length > 1500);
            if (big >= 0)
            {
                turns[big].Text = turns[big].Text[..1400] + "\n[...older tool output truncated...]\n</tool_response>";
                continue;
            }

            var lastUserTurn = turns.FindLastIndex(t => t.Role == "user" && !t.IsToolResult);
            if (lastUserTurn > 0)
            {
                turns.RemoveAt(0);
                continue;
            }
            break;
        }
        while (turns.Count > 1 && turns[0].Role == "assistant") turns.RemoveAt(0);

        // ---- assemble ChatML ---------------------------------------------
        var sb = new StringBuilder();
        sb.Append("<|im_start|>system\n").Append(systemText).Append("<|im_end|>\n");
        foreach (var t in turns)
            sb.Append("<|im_start|>").Append(t.Role).Append('\n').Append(t.Text).Append("<|im_end|>\n");
        sb.Append("<|im_start|>assistant\n");

        var prompt = sb.ToString();
        var promptTokens = Est(prompt);
        var maxTokens = Math.Clamp(_settings.ContextSize - promptTokens - 128, 512, Math.Max(512, _settings.MaxOutputTokens));

        logger.LogInformation("Prompt ≈{PromptTokens} tokens, {Turns} turns, max output {MaxTokens}", promptTokens, turns.Count, maxTokens);

        var reply = await runtime.InferTextAsync(
            ModelKind.Coder,
            prompt,
            maxTokens,
            0.2f,
            ["<|im_end|>", "<|endoftext|>", "</tool_call>"],
            cancellationToken);

        logger.LogInformation("Raw coder output length: {Length}, preview: {Preview}",
            reply?.Length ?? 0,
            string.IsNullOrEmpty(reply) ? "<EMPTY>" : reply[..Math.Min(reply.Length, 180)]);

        return reply ?? string.Empty;
    }

    // ---- helpers ---------------------------------------------------------

    private static string RenderTools(JsonElement tools)
    {
        if (tools.ValueKind != JsonValueKind.Array) return "";

        var sb = new StringBuilder("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
        foreach (var item in tools.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object) continue;
            var def = item.TryGetProperty("function", out var f) && f.ValueKind == JsonValueKind.Object ? f : item;

            var name = Str(def, "name");
            if (string.IsNullOrWhiteSpace(name)) continue;

            sb.Append("<function>\n<name>").Append(name).Append("</name>\n");
            if (Str(def, "description") is { Length: > 0 } description)
                sb.Append("<description>").Append(description).Append("</description>\n");

            sb.Append("<parameters>\n");
            JsonElement schema = default;
            foreach (var key in SchemaKeys)
            {
                if (def.TryGetProperty(key, out var s) && s.ValueKind == JsonValueKind.Object) { schema = s; break; }
            }

            if (schema.ValueKind == JsonValueKind.Object)
            {
                if (schema.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object)
                {
                    foreach (var p in props.EnumerateObject())
                    {
                        sb.Append("<parameter>\n<name>").Append(p.Name).Append("</name>\n");
                        if (Str(p.Value, "type") is { Length: > 0 } type)
                            sb.Append("<type>").Append(type).Append("</type>\n");
                        if (Str(p.Value, "description") is { Length: > 0 } pd)
                            sb.Append("<description>").Append(pd).Append("</description>\n");
                        sb.Append("</parameter>\n");
                    }
                }
                if (schema.TryGetProperty("required", out var req) && req.ValueKind == JsonValueKind.Array)
                    sb.Append("<required>").Append(req.GetRawText()).Append("</required>\n");
            }

            sb.Append("</parameters>\n</function>\n");
        }
        sb.Append("</tools>\n\n").Append(CallFormat);
        return sb.ToString();
    }

    private static string RenderCall(string? name, JsonElement args)
    {
        var sb = new StringBuilder("<tool_call>\n<function=").Append(name).Append(">\n");
        if (args.ValueKind == JsonValueKind.Object)
        {
            foreach (var p in args.EnumerateObject())
            {
                sb.Append("<parameter=").Append(p.Name).Append(">\n")
                  .Append(p.Value.ValueKind == JsonValueKind.String ? p.Value.GetString() : p.Value.GetRawText())
                  .Append("\n</parameter>\n");
            }
        }
        return sb.Append("</function>\n</tool_call>").ToString();
    }

    private static string Clip(string text, int max) =>
        text.Length <= max
            ? text
            : text[..max] + $"\n[Output truncated: {text.Length - max} more characters omitted.]";

    // Conservative token estimate (code ≈ 3 chars/token, Persian/non-ASCII ≈ 1.6 chars/token).
    private static int Est(string s)
    {
        double tokens = 0;
        foreach (var c in s) tokens += c < 128 ? 0.32 : 0.6;
        return (int)tokens + 1;
    }

    private static string? Str(JsonElement obj, string key) =>
        obj.ValueKind == JsonValueKind.Object && obj.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString()
            : null;

    private sealed class Turn(string role, string text, bool isToolResult)
    {
        public string Role { get; } = role;
        public string Text { get; set; } = text;
        public bool IsToolResult { get; } = isToolResult;
    }
}