using System.Text;
using ai_agent.Models;
using Microsoft.Extensions.Options;

namespace ai_agent.Services;

public sealed class VisionClient(ModelProcess runtime, IOptions<ModelOptions> options)
{
    private readonly ModelOptions _settings = options.Value;

    public static bool IsValidImage(VisionImage image)
    {
        if (string.IsNullOrWhiteSpace(image.Id) ||
            image.Id.Length > 120 ||
            image.Name?.Length > 255 ||
            image.DataUrl is not { Length: > 0 and <= 12_000_000 } url)
        {
            return false;
        }

        var separator = url.IndexOf(',');
        if (separator < 0) return false;

        var header = url[..separator];
        if (header is not ("data:image/png;base64" or "data:image/jpeg;base64" or "data:image/webp;base64"))
        {
            return false;
        }

        var encoded = url.AsSpan(separator + 1);
        var buffer = new byte[(encoded.Length / 4 + 1) * 3];
        if (!Convert.TryFromBase64Chars(encoded, buffer, out var length) || length == 0 || length > 10_000_000)
        {
            return false;
        }

        var bytes = buffer.AsSpan(0, length);
        return header switch
        {
            "data:image/png;base64" => bytes.StartsWith(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }),
            "data:image/jpeg;base64" => bytes.StartsWith(new byte[] { 255, 216, 255 }),
            "data:image/webp;base64" => bytes.Length >= 12 &&
                                        bytes[..4].SequenceEqual("RIFF"u8) &&
                                        bytes.Slice(8, 4).SequenceEqual("WEBP"u8),
            _ => false
        };
    }

    public static byte[] ExtractImageBytes(VisionImage image)
    {
        var url = image.DataUrl ?? throw new ArgumentException("Image DataUrl is missing.");
        var commaIndex = url.IndexOf(',');
        if (commaIndex < 0)
            throw new FormatException("Invalid image DataUrl format.");

        var base64Part = url[(commaIndex + 1)..];
        return Convert.FromBase64String(base64Part);
    }

    public async Task<string> AnalyzeAsync(IReadOnlyList<VisionImage> images, string userRequest, CancellationToken cancellationToken)
    {
        var imageBytesList = new List<(string Name, byte[] Bytes)>(images.Count);
        foreach (var img in images)
        {
            imageBytesList.Add((img.Name ?? img.Id ?? "image", ExtractImageBytes(img)));
        }

        var sb = new StringBuilder();
        sb.AppendLine("You are a visual inspection and comparison assistant for a software developer.");
        sb.AppendLine("Respond in the same language as the user's request (Persian if the user writes in Persian).");
        sb.AppendLine("Examine every attached image directly. When multiple images are provided, compare them carefully, point out differences, and explain what changes are needed in the user's project (such as Flutter/Dart or the active codebase) without forcing unrelated HTML or templates.");
        sb.AppendLine();
        sb.AppendLine($"User request: {userRequest}");
        sb.AppendLine($"Attached images ({imageBytesList.Count}):");
        for (var i = 0; i < imageBytesList.Count; i++)
        {
            sb.AppendLine($"- Image {i + 1}: {imageBytesList[i].Name}");
        }

        var prompt = $"<|im_start|>system\nYou inspect attached images directly and respond accurately in the user's language.<|im_end|>\n" +
                     $"<|im_start|>user\n{sb}<|im_end|>\n" +
                     $"<|im_start|>assistant\n";

        var result = await runtime.InferVisionAsync(
            prompt,
            imageBytesList.Select(x => x.Bytes).ToList(),
            Math.Max(512, _settings.VisionMaxOutputTokens),
            0.2f,
            ["<|im_end|>", "<|endoftext|>"],
            cancellationToken);

        if (string.IsNullOrWhiteSpace(result))
            throw new InvalidOperationException("Vision model returned an empty response.");

        return result;
    }
}