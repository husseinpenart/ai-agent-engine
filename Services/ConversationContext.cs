using System.Text;
using ai_agent.Models;
using Microsoft.Extensions.Options;

namespace ai_agent.Services;

public sealed class ConversationContext(IOptions<ModelOptions> options)
{
    public (List<ChatMessage> Recent, string Summary) Prepare(List<ChatMessage> messages)
    {
        var limit = Math.Clamp(options.Value.RecentMessages, 10, 40);
        if (messages.Count <= limit) return (messages.ToList(), "");
        var start = messages.Count - limit;
        while (start > 0 && messages[start].Role == "tool") start--;
        var latestUser = messages.FindLastIndex(start - 1, start, item => item.Role == "user");
        var recent = messages.Skip(start).ToList();
        if (recent.All(item => item.Role != "user") && latestUser >= 0) recent.Insert(0, messages[latestUser]);
        var summary = new StringBuilder();
        for (var index = Math.Max(0, start - 24); index < start && summary.Length < 1800; index++)
        {
            if (index == latestUser && recent.FirstOrDefault() == messages[index]) continue;
            var message = messages[index];
            var content = (message.Content ?? "").Replace('\r', ' ').Replace('\n', ' ').Trim();
            if (content.Length > 200) content = content[..200] + "…";
            if (content.Length > 0) summary.Append(message.Role).Append(": ").Append(content).Append('\n');
            if (message.ToolCalls is { Count: > 0 }) summary.Append("tools: ").AppendJoin(", ", message.ToolCalls.Select(call => call.Name)).Append('\n');
        }
        if (summary.Length > 2000) summary.Length = 2000;
        return (recent, summary.ToString());
    }
}
