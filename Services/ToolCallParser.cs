using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace ai_agent.Services;

/// <summary>
/// Parses tool calls from Qwen3-Coder output. Native format:
/// <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
/// Falls back to {"name":..., "arguments":{...}} JSON, but ONLY inside a <tool_call> block.
/// Incomplete (cut-off) calls are never returned, so a half-written file is never saved.
/// Uses the existing ParsedToolInvocation record (defined in WorkspaceToolRunner.cs).
/// </summary>
public static partial class ToolCallParser
{
    private const string Open = "<tool_call>";
    private const string Close = "</tool_call>";

    private static readonly JsonDocumentOptions Loose = new()
    {
        AllowTrailingCommas = true,
        CommentHandling = JsonCommentHandling.Skip
    };

    // Function must be properly closed, otherwise the call was cut off and is rejected.
    [GeneratedRegex(@"<function=([^>\s]+)\s*>(.*?)</function>", RegexOptions.Singleline)]
    private static partial Regex FunctionRegex();

    [GeneratedRegex(@"<parameter=([^>\s]+)\s*>(.*?)(?:</parameter>|(?=<parameter=)|\z)", RegexOptions.Singleline)]
    private static partial Regex ParameterRegex();

    [GeneratedRegex(@"<tool_call>.*?(?:</tool_call>|\z)", RegexOptions.Singleline)]
    private static partial Regex ToolBlockRegex();

    public static List<ParsedToolInvocation> Extract(string? output, JsonElement tools = default)
    {
        var results = new List<ParsedToolInvocation>();
        if (string.IsNullOrWhiteSpace(output)) return results;

        foreach (var block in Blocks(output))
        {
            if (TryParseXml(block, tools, out var call) || TryParseJson(block, out call))
                results.Add(call);
        }
        return results;
    }

    /// <summary>Returns the assistant text with all tool-call syntax removed.</summary>
    public static string Strip(string? output)
    {
        if (string.IsNullOrWhiteSpace(output)) return "";
        var s = ToolBlockRegex().Replace(output, "");
        var fn = s.IndexOf("<function=", StringComparison.Ordinal);
        if (fn >= 0) s = s[..fn];
        return s.Replace("<|im_end|>", "").Trim();
    }

    private static IEnumerable<string> Blocks(string output)
    {
        var found = false;
        var pos = 0;
        while (true)
        {
            var start = output.IndexOf(Open, pos, StringComparison.Ordinal);
            if (start < 0) break;
            found = true;
            var inner = start + Open.Length;
            var end = output.IndexOf(Close, inner, StringComparison.Ordinal);
            yield return (end >= 0 ? output[inner..end] : output[inner..]).Trim();
            if (end < 0) yield break;
            pos = end + Close.Length;
        }

        if (!found)
        {
            // Model sometimes forgets the <tool_call> wrapper.
            var fn = output.IndexOf("<function=", StringComparison.Ordinal);
            if (fn >= 0) yield return output[fn..];
        }
    }

    private static bool TryParseXml(string block, JsonElement tools, out ParsedToolInvocation call)
    {
        call = default!;
        var fn = FunctionRegex().Match(block);
        if (!fn.Success) return false;

        var name = fn.Groups[1].Value.Trim();
        if (name.Length == 0) return false;

        var args = new Dictionary<string, object?>();
        foreach (Match p in ParameterRegex().Matches(fn.Groups[2].Value))
        {
            var key = p.Groups[1].Value.Trim();
            if (key.Length == 0) continue;
            args[key] = Coerce(SchemaType(tools, name, key), TrimOuterNewline(p.Groups[2].Value));
        }

        call = new ParsedToolInvocation(NewId(), name, JsonSerializer.SerializeToElement(args));
        return true;
    }

    private static bool TryParseJson(string block, out ParsedToolInvocation call)
    {
        call = default!;
        var candidate = FirstBalancedObject(block);
        if (candidate is null) return false;

        try
        {
            using var doc = JsonDocument.Parse(candidate, Loose);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) return false;

            var name = Str(root, "name") ?? Str(root, "tool");
            if (string.IsNullOrWhiteSpace(name)) return false;

            var args = JsonSerializer.SerializeToElement(new { });
            if (root.TryGetProperty("arguments", out var a) || root.TryGetProperty("args", out a))
            {
                if (a.ValueKind == JsonValueKind.Object)
                {
                    args = a.Clone();
                }
                else if (a.ValueKind == JsonValueKind.String && FirstBalancedObject(a.GetString() ?? "") is { } inner)
                {
                    using var innerDoc = JsonDocument.Parse(inner, Loose);
                    args = innerDoc.RootElement.Clone();
                }
            }

            var id = Str(root, "id");
            call = new ParsedToolInvocation(string.IsNullOrWhiteSpace(id) ? NewId() : id.Trim(), name.Trim(), args);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    // Returns the first complete {...} object, or null when it is unbalanced (= cut off).
    private static string? FirstBalancedObject(string raw)
    {
        var start = raw.IndexOf('{');
        if (start < 0) return null;

        var sb = new StringBuilder();
        bool inStr = false, esc = false;
        var depth = 0;

        for (var i = start; i < raw.Length; i++)
        {
            var ch = raw[i];
            if (esc) { esc = false; sb.Append(ch); continue; }

            if (inStr)
            {
                if (ch == '\\') { esc = true; sb.Append(ch); }
                else if (ch == '"') { inStr = false; sb.Append(ch); }
                else if (ch == '\n') sb.Append("\\n");
                else if (ch == '\r') sb.Append("\\r");
                else if (ch == '\t') sb.Append("\\t");
                else sb.Append(ch);
                continue;
            }

            sb.Append(ch);
            if (ch == '"') inStr = true;
            else if (ch == '{') depth++;
            else if (ch == '}' && --depth == 0) return sb.ToString();
        }
        return null;
    }

    private static string TrimOuterNewline(string value)
    {
        if (value.StartsWith("\r\n", StringComparison.Ordinal)) value = value[2..];
        else if (value.StartsWith('\n')) value = value[1..];

        if (value.EndsWith("\r\n", StringComparison.Ordinal)) value = value[..^2];
        else if (value.EndsWith('\n')) value = value[..^1];
        return value;
    }

    // Values stay strings unless the tool schema explicitly says otherwise (so file contents are never altered).
    private static object? Coerce(string? type, string raw)
    {
        var t = raw.Trim();
        if (type == "integer" && long.TryParse(t, NumberStyles.Integer, CultureInfo.InvariantCulture, out var l)) return l;
        if (type == "number" && double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)) return d;
        if (type == "boolean" && bool.TryParse(t, out var b)) return b;
        if (type is "object" or "array")
        {
            try
            {
                using var doc = JsonDocument.Parse(raw, Loose);
                return doc.RootElement.Clone();
            }
            catch (JsonException) { }
        }
        return raw;
    }

    private static string? SchemaType(JsonElement tools, string tool, string param)
    {
        if (tools.ValueKind != JsonValueKind.Array) return null;

        foreach (var item in tools.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object) continue;
            var def = item.TryGetProperty("function", out var f) && f.ValueKind == JsonValueKind.Object ? f : item;
            if (!string.Equals(Str(def, "name"), tool, StringComparison.Ordinal)) continue;

            foreach (var key in new[] { "parameters", "input_schema", "inputSchema" })
            {
                if (def.TryGetProperty(key, out var schema) && schema.ValueKind == JsonValueKind.Object &&
                    schema.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object &&
                    props.TryGetProperty(param, out var p) && p.ValueKind == JsonValueKind.Object)
                {
                    return Str(p, "type");
                }
            }
        }
        return null;
    }

    private static string? Str(JsonElement obj, string key) =>
        obj.ValueKind == JsonValueKind.Object && obj.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString()
            : null;

    private static string NewId() => "call_" + Guid.NewGuid().ToString("N")[..8];
}