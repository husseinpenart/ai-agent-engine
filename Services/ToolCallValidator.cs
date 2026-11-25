using System.Text.Json;

namespace ai_agent.Services;

public sealed record PendingToolCall(int Index, string Id, string Name, string Arguments);
public sealed record ValidatedToolCall(string Id, string Name, JsonElement Arguments, string? Error);

public static class ToolCallValidator
{
    public static List<ValidatedToolCall> ParseBatch(IEnumerable<PendingToolCall> pending, bool interrupted, bool outOfOrder)
    {
        var calls = pending.OrderBy(call => call.Index).ToList();
        var highestIndex = calls.Count == 0 ? -1 : calls[^1].Index;
        var validated = new List<ValidatedToolCall>();
        var blocked = false;

        foreach (var call in calls)
        {
            var isLastOrOutOfOrder = interrupted && (outOfOrder || call.Index == highestIndex);
            var (args, error) = Parse(call.Name, call.Arguments, isLastOrOutOfOrder);

            if (string.IsNullOrWhiteSpace(call.Name))
            {
                error = "The tool call has no name.";
            }
            else if (blocked)
            {
                error = "An earlier tool call in the batch was incomplete or invalid; retry sequentially after checking previous results.";
            }

            if (error is not null)
            {
                blocked = true;
            }

            validated.Add(new ValidatedToolCall(call.Id, call.Name.Trim(), args, error));
        }

        return validated;
    }

    public static (JsonElement Arguments, string? Error) Parse(string name, string arguments, bool interrupted)
    {
        var empty = JsonSerializer.SerializeToElement(new { });

        JsonElement parsed;
        try
        {
            var rawJson = string.IsNullOrWhiteSpace(arguments) ? "{}" : arguments.Trim();
            using var document = JsonDocument.Parse(rawJson);
            parsed = document.RootElement.Clone();
        }
        catch (JsonException)
        {
            return (empty, interrupted
                ? "The model response was cut off before tool arguments completed; JSON is incomplete."
                : "Tool arguments are incomplete or invalid JSON.");
        }

        if (parsed.ValueKind != JsonValueKind.Object)
        {
            return (empty, "Tool arguments must be a JSON object.");
        }

        if (interrupted)
        {
            return (empty, "The model response was cut off before the tool call finished; retry with complete JSON.");
        }

        if (name is not ("write_file" or "append_file"))
        {
            return (parsed, null);
        }

        if (!parsed.TryGetProperty("path", out var path) ||
            path.ValueKind != JsonValueKind.String ||
            string.IsNullOrWhiteSpace(path.GetString()) ||
            !parsed.TryGetProperty("content", out var content) ||
            content.ValueKind != JsonValueKind.String)
        {
            return (empty, "File tools require a nonempty path and a string content field.");
        }

        return (parsed, null);
    }
}