using System.Text;
using ai_agent.Models;
using Microsoft.Extensions.Options;

namespace ai_agent.Services;

public sealed class ConversationContext(IOptions<ModelOptions> options)
{
    public (List<ChatMessage> Recent, string Summary) Prepare(List<ChatMessage> messages)
    {
        var limit = Math.Clamp(options.Value.RecentMessages, 4, 40);
        var toolLimit = Math.Clamp(options.Value.ToolResultMaxChars, 256, 10000);
        if (messages.Count <= limit) return (LimitToolResults(messages, toolLimit), "");
        var start = messages.Count - limit;
        while (start > 0 && messages[start].Role == "tool") start--;
        var latestUser = messages.FindLastIndex(start - 1, start, item => item.Role == "user");
        var recent = messages.Skip(start).ToList();
        if (recent.All(item => item.Role != "user") && latestUser >= 0) recent.Insert(0, messages[latestUser]);
        var summary = new StringBuilder();
        for (var index = Math.Max(0, start - 12); index < start && summary.Length < 600; index++)
        {
            if (index == latestUser && recent.FirstOrDefault() == messages[index]) continue;
            var message = messages[index];
            var content = (message.Content ?? "").Replace('\r', ' ').Replace('\n', ' ').Trim();
            if (content.Length > 120) content = content[..120] + "…";
            if (content.Length > 0) summary.Append(message.Role).Append(": ").Append(content).Append('\n');
            if (message.ToolCalls is { Count: > 0 }) summary.Append("tools: ").AppendJoin(", ", message.ToolCalls.Select(call => call.Name)).Append('\n');
        }
        if (summary.Length > 800) summary.Length = 800;
        return (LimitToolResults(recent, toolLimit), summary.ToString());
    }

    private static List<ChatMessage> LimitToolResults(IEnumerable<ChatMessage> messages, int maxChars) => messages.Select(message =>
        message.Role == "tool" && message.Content is { } content && content.Length > maxChars
            ? message with { Content = content[..maxChars] + "\n[Tool output truncated]" }
            : message).ToList();
}
