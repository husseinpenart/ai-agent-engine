using System.Text.Json;

namespace ai_agent.Services;

public sealed record PendingToolCall(int Index, string Id, string Name, string Arguments);
public sealed record ValidatedToolCall(string Id, string Name, JsonElement Arguments, string? Error);

public static class ToolCallValidator
{
    public const int MaxFileChunkCharacters = 6000;

    public static List<ValidatedToolCall> ParseBatch(IEnumerable<PendingToolCall> pending, bool interrupted, bool outOfOrder)
    {
        var calls = pending.OrderBy(call => call.Index).ToList();
        var highestIndex = calls.Count == 0 ? -1 : calls[^1].Index;
        var validated = new List<ValidatedToolCall>();
        var blocked = false;
        foreach (var call in calls)
        {
            var (args, error) = Parse(call.Name, call.Arguments, interrupted && (outOfOrder || call.Index == highestIndex));
            if (string.IsNullOrEmpty(call.Name)) error = "The tool call has no name.";
            else if (blocked) error = "A previous call was incomplete or invalid, so this call was not executed.";
            if (error is not null) blocked = true;
            validated.Add(new ValidatedToolCall(call.Id, call.Name, args, error));
        }
        return validated;
    }

    public static (JsonElement Arguments, string? Error) Parse(string name, string arguments, bool interrupted)
    {
        var empty = JsonSerializer.SerializeToElement(new { });
        if (interrupted) return (empty, "The model response was cut off before all tool calls completed.");
        JsonElement parsed;
        try
        {
            using var document = JsonDocument.Parse(string.IsNullOrWhiteSpace(arguments) ? "{}" : arguments);
            parsed = document.RootElement.Clone();
        }
        catch (JsonException) { return (empty, "Tool arguments are incomplete or invalid JSON."); }
        if (parsed.ValueKind != JsonValueKind.Object) return (empty, "Tool arguments must be a JSON object.");
        if (name is not ("write_file" or "append_file")) return (parsed, null);
        if (!parsed.TryGetProperty("path", out var path) || path.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(path.GetString()) ||
            !parsed.TryGetProperty("content", out var content) || content.ValueKind != JsonValueKind.String)
            return (empty, "File tools require a nonempty path and a string content field.");
        if (content.GetString()!.Length > MaxFileChunkCharacters)
            return (JsonSerializer.SerializeToElement(new { path = path.GetString() }), $"File chunk exceeds {MaxFileChunkCharacters} characters.");
        return (parsed, null);
    }
}
