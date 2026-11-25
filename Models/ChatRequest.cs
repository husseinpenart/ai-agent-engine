using System.Text.Json;

namespace ai_agent.Models;

public sealed record ChatRequest(
    string? SessionId,
    List<ChatMessage>? Messages,
    List<ToolDefinition>? Tools,
    JsonElement Context,
    List<VisionImage>? Images = null);

public sealed record VisionImage(string? Id, string? Name, string? DataUrl);

public sealed record ChatMessage(
    string? Role,
    string? Content,
    List<ToolCall>? ToolCalls,
    string? ToolCallId,
    string? Name);

public sealed record ToolCall(string? Id, string? Name, JsonElement Args);

public sealed record ToolDefinition(string? Name, string? Description, JsonElement Parameters);