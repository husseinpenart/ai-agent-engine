namespace ai_agent.Services;

/// <summary>
/// Replaces old_str with new_str in file content the way a coding agent needs:
///  - ignores CRLF vs LF differences (restores the file's original line endings)
///  - refuses ambiguous matches
///  - falls back to a line-by-line match that ignores indentation/trailing whitespace
///  - returns an error message that tells the model what to do next
/// Port the same logic to the Tauri side if edit_file runs there.
/// </summary>
public static class EditMatcher
{
    public static bool TryReplace(string content, string oldStr, string newStr, out string updated, out string error)
    {
        updated = content;
        error = "";

        if (string.IsNullOrEmpty(oldStr))
        {
            error = "old_str is empty. Use write_file to create or fully overwrite a file.";
            return false;
        }

        var crlf = content.Contains("\r\n");
        var text = content.Replace("\r\n", "\n");
        var oldN = oldStr.Replace("\r\n", "\n");
        var newN = newStr.Replace("\r\n", "\n");

        // 1) exact match
        var first = text.IndexOf(oldN, StringComparison.Ordinal);
        if (first >= 0)
        {
            if (text.IndexOf(oldN, first + oldN.Length, StringComparison.Ordinal) >= 0)
            {
                error = "old_str matches more than once. Include more surrounding lines so it is unique.";
                return false;
            }

            updated = Restore(text[..first] + newN + text[(first + oldN.Length)..], crlf);
            return true;
        }

        // 2) whitespace-tolerant, line based
        var lines = text.Split('\n');
        var oldLines = oldN.Trim('\n').Split('\n');
        var found = -1;
        var count = 0;

        for (var i = 0; i + oldLines.Length <= lines.Length; i++)
        {
            var ok = true;
            for (var j = 0; j < oldLines.Length && ok; j++)
                ok = lines[i + j].Trim() == oldLines[j].Trim();

            if (ok)
            {
                if (found < 0) found = i;
                count++;
            }
        }

        if (count == 1)
        {
            var result = lines[..found]
                .Concat(newN.Trim('\n').Split('\n'))
                .Concat(lines[(found + oldLines.Length)..]);
            updated = Restore(string.Join('\n', result), crlf);
            return true;
        }

        error = count > 1
            ? "old_str matches multiple places (ignoring whitespace). Include more surrounding lines."
            : "old_str was not found. The file may have changed or the text differs. Call read_file again and copy the exact text.";
        return false;
    }

    private static string Restore(string s, bool crlf) => crlf ? s.Replace("\n", "\r\n") : s;
}