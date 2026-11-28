using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace ai_agent.Services;

public sealed record ParsedToolInvocation(string Id, string Name, JsonElement Args);

public sealed partial class WorkspaceToolRunner(ILogger<WorkspaceToolRunner> logger)
{
    private static readonly HashSet<string> IgnoredDirs = new(StringComparer.OrdinalIgnoreCase)
    {
        ".git", ".dart_tool", ".idea", ".vscode", "build", "node_modules", "bin", "obj", ".gradle", "dist", "target", "venv", "__pycache__"
    };

    private static readonly HashSet<string> MetaKeys = new(StringComparer.OrdinalIgnoreCase)
    {
        "id", "name", "tool", "type", "description", "arguments", "args"
    };

    private static readonly JsonDocumentOptions LooseJsonOptions = new()
    {
        AllowTrailingCommas = true,
        CommentHandling = JsonCommentHandling.Skip
    };

    [GeneratedRegex(@"\[(?:Executed\s+)?Tool calls:\s*(\[[\s\S]*)", RegexOptions.IgnoreCase)]
    private static partial Regex BracketToolCallsRegex();

    [GeneratedRegex(@"^\s*([a-zA-Z0-9_]+)\s*\(\s*(.*?)\s*\)\s*;?\s*$", RegexOptions.Singleline)]
    private static partial Regex FunctionLikeCallRegex();

    public static string? ResolveRoot(JsonElement context)
    {
        if (context.ValueKind != JsonValueKind.Object)
            return null;

        string?[] keys = ["workspaceRoot", "projectPath", "rootPath", "path", "cwd", "activeWorkspace", "workspace"];
        foreach (var key in keys)
        {
            if (context.TryGetProperty(key, out var prop) && prop.ValueKind == JsonValueKind.String)
            {
                var val = prop.GetString()?.Trim();
                if (!string.IsNullOrWhiteSpace(val) && Directory.Exists(val))
                    return Path.GetFullPath(val);
            }
        }

        if (context.TryGetProperty("workspaces", out var ws) && ws.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in ws.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.Object)
                {
                    var isEnabled = !item.TryGetProperty("enabled", out var enProp) || enProp.ValueKind != JsonValueKind.False;
                    if (!isEnabled) continue;

                    foreach (var key in keys)
                    {
                        if (item.TryGetProperty(key, out var p) && p.ValueKind == JsonValueKind.String)
                        {
                            var val = p.GetString()?.Trim();
                            if (!string.IsNullOrWhiteSpace(val) && Directory.Exists(val))
                                return Path.GetFullPath(val);
                        }
                    }
                }
                else if (item.ValueKind == JsonValueKind.String)
                {
                    var val = item.GetString()?.Trim();
                    if (!string.IsNullOrWhiteSpace(val) && Directory.Exists(val))
                        return Path.GetFullPath(val);
                }
            }
        }

        return null;
    }

    public static List<ParsedToolInvocation> ExtractToolCalls(string modelOutput)
    {
        var results = new List<ParsedToolInvocation>();
        if (string.IsNullOrWhiteSpace(modelOutput))
            return results;

        const string openTag = "<tool_call>";
        const string closeTag = "</tool_call>";

        var pos = 0;
        var idx = 0;
        while (pos < modelOutput.Length)
        {
            var start = modelOutput.IndexOf(openTag, pos, StringComparison.Ordinal);
            if (start < 0) break;

            var afterOpen = start + openTag.Length;
            var end = modelOutput.IndexOf(closeTag, afterOpen, StringComparison.Ordinal);

            var rawInner = end >= 0
                ? modelOutput[afterOpen..end].Trim()
                : modelOutput[afterOpen..].Trim();

            pos = end >= 0 ? end + closeTag.Length : modelOutput.Length;

            if (TryParseSingleInvocation(rawInner, ref idx, out var parsed))
                results.Add(parsed);
        }

        if (results.Count > 0)
            return results;

        var match = BracketToolCallsRegex().Match(modelOutput);
        if (match.Success && TryParseJsonArrayLoosely(match.Groups[1].Value.Trim(), out var arrElement))
        {
            foreach (var item in arrElement.EnumerateArray())
            {
                if (TryParseInvocationElement(item, ref idx, out var parsed))
                    results.Add(parsed);
            }
        }

        return results;
    }

    private static bool TryParseSingleInvocation(string rawBlock, ref int idx, out ParsedToolInvocation parsed)
    {
        parsed = default!;
        if (string.IsNullOrWhiteSpace(rawBlock))
            return false;

        if (TryParseJsonObjectLoosely(rawBlock, out var objElement) &&
            TryParseInvocationElement(objElement, ref idx, out parsed))
        {
            return true;
        }

        var fnMatch = FunctionLikeCallRegex().Match(rawBlock);
        if (!fnMatch.Success)
            return false;

        var toolName = fnMatch.Groups[1].Value.Trim();
        var insideParens = fnMatch.Groups[2].Value.Trim();
        if (string.IsNullOrWhiteSpace(toolName))
            return false;

        var id = $"call_{++idx}";
        Dictionary<string, object?> argsDict;

        if (string.IsNullOrEmpty(insideParens))
        {
            argsDict = new Dictionary<string, object?>();
        }
        else if (TryParseJsonArrayLoosely($"[{insideParens}]", out var arrArgs) && arrArgs.ValueKind == JsonValueKind.Array)
        {
            var positional = arrArgs.EnumerateArray()
                .Select(e => e.ValueKind == JsonValueKind.String ? e.GetString() : e.ToString())
                .ToArray();
            argsDict = MapPositionalArgs(toolName, positional);
        }
        else
        {
            var trimmedArg = insideParens.Trim('"', '\'', '`');
            argsDict = MapPositionalArgs(toolName, [trimmedArg]);
        }

        var argsElement = JsonSerializer.SerializeToElement(argsDict);
        parsed = new ParsedToolInvocation(id, toolName, argsElement);
        return true;
    }

    private static Dictionary<string, object?> MapPositionalArgs(string toolName, string?[] positional)
    {
        var lower = toolName.Trim().ToLowerInvariant();
        var dict = new Dictionary<string, object?>();

        if (positional.Length == 0 || positional[0] is null)
            return dict;

        switch (lower)
        {
            case "search_files":
            case "list_files":
            case "list_directory":
                dict["query"] = positional[0];
                if (positional.Length > 1 && positional[1] is not null)
                    dict["path"] = positional[1];
                break;

            case "read_file":
            case "delete_path":
            case "create_directory":
                dict["path"] = positional[0];
                break;

            case "write_file":
            case "append_file":
                dict["path"] = positional[0];
                if (positional.Length > 1 && positional[1] is not null)
                    dict["content"] = positional[1];
                break;

            case "edit_file":
            case "replace_in_file":
                dict["path"] = positional[0];
                if (positional.Length > 1 && positional[1] is not null)
                    dict["old_str"] = positional[1];
                if (positional.Length > 2 && positional[2] is not null)
                    dict["new_str"] = positional[2];
                break;

            case "run_command":
                dict["command"] = positional[0];
                break;

            default:
                dict["path"] = positional[0];
                break;
        }

        return dict;
    }

    private static bool TryParseJsonObjectLoosely(string raw, out JsonElement element)
    {
        element = default;
        foreach (var candidate in BuildJsonCandidates(raw, '{', '}'))
        {
            try
            {
                using var doc = JsonDocument.Parse(candidate, LooseJsonOptions);
                if (doc.RootElement.ValueKind == JsonValueKind.Object)
                {
                    element = doc.RootElement.Clone();
                    return true;
                }
            }
            catch (JsonException)
            {
            }
        }
        return false;
    }

    private static bool TryParseJsonArrayLoosely(string raw, out JsonElement element)
    {
        element = default;
        foreach (var candidate in BuildJsonCandidates(raw, '[', ']'))
        {
            try
            {
                using var doc = JsonDocument.Parse(candidate, LooseJsonOptions);
                if (doc.RootElement.ValueKind == JsonValueKind.Array)
                {
                    element = doc.RootElement.Clone();
                    return true;
                }
            }
            catch (JsonException)
            {
            }
        }
        return false;
    }

    private static IEnumerable<string> BuildJsonCandidates(string raw, char openChar, char closeChar)
    {
        var trimmed = raw.Trim();
        var startIdx = trimmed.IndexOf(openChar);
        if (startIdx < 0) yield break;

        var core = trimmed[startIdx..];
        var sb = new StringBuilder(core.Length + 16);
        var inString = false;
        var escaped = false;
        var braceDepth = 0;
        var bracketDepth = 0;
        var closedAt = -1;

        for (var i = 0; i < core.Length; i++)
        {
            var ch = core[i];
            if (escaped)
            {
                escaped = false;
                sb.Append(ch);
                continue;
            }
            if (ch == '\\' && inString)
            {
                escaped = true;
                sb.Append(ch);
                continue;
            }
            if (ch == '"')
            {
                inString = !inString;
                sb.Append(ch);
                continue;
            }
            if (inString && ch < 0x20)
            {
                if (ch == '\n') sb.Append("\\n");
                else if (ch == '\r') sb.Append("\\r");
                else if (ch == '\t') sb.Append("\\t");
                continue;
            }
            if (!inString)
            {
                if (ch == '{') braceDepth++;
                else if (ch == '}') braceDepth--;
                else if (ch == '[') bracketDepth++;
                else if (ch == ']') bracketDepth--;

                sb.Append(ch);

                if (openChar == '{' && ch == '}' && braceDepth == 0 && bracketDepth == 0)
                {
                    closedAt = sb.Length;
                    break;
                }
                if (openChar == '[' && ch == ']' && bracketDepth == 0 && braceDepth == 0)
                {
                    closedAt = sb.Length;
                    break;
                }
                continue;
            }
            sb.Append(ch);
        }

        if (closedAt > 0)
        {
            yield return sb.ToString()[..closedAt];
        }

        if (escaped) sb.Append('\\');
        if (inString) sb.Append('"');
        while (braceDepth > 0) { sb.Append('}'); braceDepth--; }
        while (bracketDepth > 0) { sb.Append(']'); bracketDepth--; }

        var repaired = sb.ToString();
        if (repaired != core)
            yield return repaired;
    }

    private static bool TryParseInvocationElement(JsonElement root, ref int idx, out ParsedToolInvocation parsed)
    {
        parsed = default!;
        if (root.ValueKind != JsonValueKind.Object)
            return false;

        string? name = null;
        if (root.TryGetProperty("name", out var nProp) && nProp.ValueKind == JsonValueKind.String)
            name = nProp.GetString()?.Trim();
        else if (root.TryGetProperty("tool", out var tProp) && tProp.ValueKind == JsonValueKind.String)
            name = tProp.GetString()?.Trim();

        if (string.IsNullOrWhiteSpace(name))
            return false;

        var id = root.TryGetProperty("id", out var idProp) && idProp.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(idProp.GetString())
            ? idProp.GetString()!.Trim()
            : $"call_{++idx}";

        JsonElement argsElement;
        if (root.TryGetProperty("arguments", out var aProp))
        {
            if (aProp.ValueKind == JsonValueKind.String)
            {
                var rawArgs = aProp.GetString();
                if (string.IsNullOrWhiteSpace(rawArgs))
                    return false;
                if (!TryParseJsonObjectLoosely(rawArgs, out argsElement))
                    return false;
            }
            else if (aProp.ValueKind == JsonValueKind.Object)
            {
                argsElement = aProp.Clone();
            }
            else
            {
                return false;
            }
        }
        else if (root.TryGetProperty("args", out var argsAlt))
        {
            if (argsAlt.ValueKind == JsonValueKind.Object)
            {
                argsElement = argsAlt.Clone();
            }
            else if (argsAlt.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(argsAlt.GetString()))
            {
                if (!TryParseJsonObjectLoosely(argsAlt.GetString()!, out argsElement))
                    return false;
            }
            else
            {
                argsElement = JsonSerializer.SerializeToElement(new { });
            }
        }
        else
        {
            var extraProps = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
            foreach (var prop in root.EnumerateObject())
            {
                if (!MetaKeys.Contains(prop.Name))
                {
                    extraProps[prop.Name] = prop.Value.Clone();
                }
            }

            argsElement = extraProps.Count > 0
                ? JsonSerializer.SerializeToElement(extraProps)
                : JsonSerializer.SerializeToElement(new { });
        }

        parsed = new ParsedToolInvocation(id, name, argsElement);
        return true;
    }

    public static string StripToolCallTags(string modelOutput)
    {
        if (string.IsNullOrWhiteSpace(modelOutput))
            return string.Empty;

        const string openTag = "<tool_call>";
        const string closeTag = "</tool_call>";

        var sb = new StringBuilder(modelOutput.Length);
        var pos = 0;
        while (pos < modelOutput.Length)
        {
            var start = modelOutput.IndexOf(openTag, pos, StringComparison.Ordinal);
            if (start < 0)
            {
                sb.Append(modelOutput[pos..]);
                break;
            }

            sb.Append(modelOutput[pos..start]);
            var end = modelOutput.IndexOf(closeTag, start + openTag.Length, StringComparison.Ordinal);
            if (end < 0)
            {
                break;
            }
            pos = end + closeTag.Length;
        }

        return sb.ToString().Trim();
    }

    public async Task<string> ExecuteAsync(ParsedToolInvocation call, string? workspaceRoot, CancellationToken cancellationToken)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(workspaceRoot) || !Directory.Exists(workspaceRoot))
            {
                return "Error: No valid project path found in request Context.";
            }

            var toolName = call.Name.Trim().ToLowerInvariant();
            switch (toolName)
            {
                case "list_files":
                case "list_directory":
                case "search_files":
                    {
                        var pattern = GetString(call.Args, "pattern") ?? GetString(call.Args, "query") ?? "*";
                        var subPath = GetString(call.Args, "path") ?? GetString(call.Args, "directory") ?? "";
                        var targetDir = ResolveSafePath(workspaceRoot, subPath);
                        if (targetDir is null || !Directory.Exists(targetDir))
                            return $"Error: Directory not found: {subPath}";

                        var files = EnumerateFilesSafe(targetDir)
                            .Select(p => Path.GetRelativePath(workspaceRoot, p).Replace('\\', '/'))
                            .Where(rel => MatchesPattern(rel, pattern))
                            .Take(400)
                            .ToList();

                        return files.Count == 0
                            ? $"No matching files found for pattern '{pattern}'."
                            : string.Join("\n", files);
                    }

                case "read_file":
                    {
                        var relPath = GetString(call.Args, "path") ?? GetString(call.Args, "filePath");
                        if (string.IsNullOrWhiteSpace(relPath))
                            return "Error: 'path' is required for read_file.";

                        var fullPath = ResolveSafePath(workspaceRoot, relPath);
                        if (fullPath is null || !File.Exists(fullPath))
                            return $"Error: File not found: {relPath}";

                        return await File.ReadAllTextAsync(fullPath, cancellationToken);
                    }

                case "write_file":
                    {
                        var relPath = GetString(call.Args, "path") ?? GetString(call.Args, "filePath");
                        var content = GetString(call.Args, "content");
                        if (string.IsNullOrWhiteSpace(relPath) || content is null)
                            return "Error: 'path' and 'content' are required for write_file.";

                        var fullPath = ResolveSafePath(workspaceRoot, relPath);
                        if (fullPath is null)
                            return $"Error: Invalid or out-of-workspace path: {relPath}";

                        var dir = Path.GetDirectoryName(fullPath);
                        if (!string.IsNullOrEmpty(dir))
                            Directory.CreateDirectory(dir);

                        await File.WriteAllTextAsync(fullPath, content, new UTF8Encoding(false), cancellationToken);
                        logger.LogInformation("Wrote file {Path}", fullPath);
                        return $"Successfully wrote {content.Length} chars to {relPath}.";
                    }

                case "append_file":
                    {
                        var relPath = GetString(call.Args, "path") ?? GetString(call.Args, "filePath");
                        var content = GetString(call.Args, "content");
                        if (string.IsNullOrWhiteSpace(relPath) || content is null)
                            return "Error: 'path' and 'content' are required for append_file.";

                        var fullPath = ResolveSafePath(workspaceRoot, relPath);
                        if (fullPath is null || !File.Exists(fullPath))
                            return $"Error: File not found for append_file: {relPath}";

                        await File.AppendAllTextAsync(fullPath, content, new UTF8Encoding(false), cancellationToken);
                        logger.LogInformation("Appended to file {Path}", fullPath);
                        return $"Successfully appended {content.Length} chars to {relPath}.";
                    }

                case "replace_in_file":
                case "edit_file":
                    {
                        var relPath = GetString(call.Args, "path") ?? GetString(call.Args, "filePath");
                        var oldText = GetString(call.Args, "old_str") ?? GetString(call.Args, "oldText") ?? GetString(call.Args, "search");
                        var newText = GetString(call.Args, "new_str") ?? GetString(call.Args, "newText") ?? GetString(call.Args, "replace");
                        if (string.IsNullOrWhiteSpace(relPath) || oldText is null || newText is null)
                            return "Error: 'path', 'old_str', and 'new_str' are required for edit_file.";

                        var fullPath = ResolveSafePath(workspaceRoot, relPath);
                        if (fullPath is null || !File.Exists(fullPath))
                            return $"Error: File not found: {relPath}";

                        var current = await File.ReadAllTextAsync(fullPath, cancellationToken);
                        var idx = current.IndexOf(oldText, StringComparison.Ordinal);
                        if (idx < 0)
                            return $"Error: Target text block not found in {relPath}.";

                        var updated = string.Concat(current.AsSpan(0, idx), newText, current.AsSpan(idx + oldText.Length));
                        await File.WriteAllTextAsync(fullPath, updated, new UTF8Encoding(false), cancellationToken);
                        logger.LogInformation("Updated file {Path}", fullPath);
                        return $"Successfully updated {relPath}.";
                    }

                default:
                    return $"Error: Unsupported tool '{call.Name}'.";
            }
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Tool execution failed for {Tool}", call.Name);
            return $"Error executing {call.Name}: {ex.Message}";
        }
    }

    private static string? GetString(JsonElement obj, string key)
    {
        if (obj.ValueKind != JsonValueKind.Object) return null;
        return obj.TryGetProperty(key, out var prop) && prop.ValueKind == JsonValueKind.String
            ? prop.GetString()
            : null;
    }

    private static string? ResolveSafePath(string root, string relativeOrSub)
    {
        var fullRoot = Path.GetFullPath(root);
        var raw = (relativeOrSub ?? string.Empty).Trim().Replace('/', Path.DirectorySeparatorChar);

        string combined;
        if (Path.IsPathRooted(raw))
        {
            combined = Path.GetFullPath(raw);
            if (!combined.StartsWith(fullRoot, StringComparison.OrdinalIgnoreCase))
                return null;
        }
        else
        {
            var normalizedSub = raw.TrimStart(Path.DirectorySeparatorChar);
            combined = Path.GetFullPath(Path.Combine(fullRoot, normalizedSub));
            if (!combined.StartsWith(fullRoot, StringComparison.OrdinalIgnoreCase))
                return null;
        }

        return combined;
    }

    private static IEnumerable<string> EnumerateFilesSafe(string startDir)
    {
        var stack = new Stack<string>();
        stack.Push(startDir);

        while (stack.Count > 0)
        {
            var current = stack.Pop();
            string[] subDirs;
            try { subDirs = Directory.GetDirectories(current); }
            catch { continue; }

            foreach (var d in subDirs)
            {
                var name = Path.GetFileName(d);
                if (!IgnoredDirs.Contains(name))
                    stack.Push(d);
            }

            string[] files;
            try { files = Directory.GetFiles(current); }
            catch { continue; }

            foreach (var f in files)
                yield return f;
        }
    }

    private static bool MatchesPattern(string relativePath, string pattern)
    {
        if (string.IsNullOrWhiteSpace(pattern) || pattern == "*" || pattern == "**")
            return true;

        var clean = pattern.Trim().TrimStart('*').ToLowerInvariant();
        if (string.IsNullOrEmpty(clean))
            return true;

        return relativePath.ToLowerInvariant().Contains(clean);
    }
}