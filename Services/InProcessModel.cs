using System.Runtime.CompilerServices;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using Microsoft.Extensions.Options;

namespace ai_agent.Services;

public sealed class InProcessModel(IOptions<ModelOptions> options, ILogger<InProcessModel> logger) : IDisposable
{
    private readonly ModelOptions settings = options.Value;
    private readonly object sync = new();
    private LLamaWeights? weights;
    private ModelKind? kind;
    private Timer? idleTimer;
    private bool active;
    private bool disposed;

    public async IAsyncEnumerable<string> GenerateAsync(
        ModelKind requested, string prompt, IReadOnlyList<byte[]>? images,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var parameters = Parameters(requested);
        LLamaWeights current;
        lock (sync)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            idleTimer?.Dispose();
            idleTimer = null;
            active = true;
            if (weights is null || kind != requested)
            {
                weights?.Dispose();
                weights = null;
                kind = null;
                if (!File.Exists(parameters.ModelPath)) throw new FileNotFoundException("Model file not found", parameters.ModelPath);
                logger.LogInformation("Loading {ModelKind} model in process with {Layers} GPU layers", requested, parameters.GpuLayerCount);
                weights = LLamaWeights.LoadFromFile(parameters);
                kind = requested;
            }
            current = weights;
        }

        using var context = current.CreateContext(parameters);
        MtmdWeights? vision = null;
        try
        {
            InteractiveExecutor executor;
            if (images is { Count: > 0 })
            {
                if (!File.Exists(settings.VisionMmprojPath)) throw new FileNotFoundException("Vision projector not found", settings.VisionMmprojPath);
                var visionParams = MtmdContextParams.Default();
                visionParams.UseGpu = true;
                visionParams.NThreads = settings.Threads;
                visionParams.ImageMaxTokens = 1024;
                vision = MtmdWeights.LoadFromFile(settings.VisionMmprojPath, current, visionParams);
                if (!vision.SupportsVision) throw new InvalidOperationException("The projector does not support image input.");
                foreach (var image in images) vision.LoadMedia(image);
                executor = new InteractiveExecutor(context, vision);
            }
            else executor = new InteractiveExecutor(context);

            var inference = new InferenceParams
            {
                MaxTokens = requested == ModelKind.Vision ? settings.VisionMaxOutputTokens : settings.MaxOutputTokens,
                AntiPrompts = ["<|im_end|>"],
                SamplingPipeline = new DefaultSamplingPipeline { Temperature = 0.2f }
            };
            await foreach (var part in executor.InferAsync(prompt, inference, cancellationToken)) yield return part;
        }
        finally { vision?.Dispose(); }
    }

    private ModelParams Parameters(ModelKind requested) => new(
        Environment.ExpandEnvironmentVariables(requested == ModelKind.Vision ? settings.VisionModelPath : settings.ModelPath))
    {
        ContextSize = (uint)(requested == ModelKind.Vision ? settings.VisionContextSize : settings.ContextSize),
        GpuLayerCount = requested == ModelKind.Vision ? settings.VisionGpuLayers :
            int.TryParse(settings.GpuLayers, out var layers) ? Math.Clamp(layers, 0, 99) : 12,
        Threads = settings.Threads,
        BatchSize = (uint)(requested == ModelKind.Vision ? settings.VisionBatchSize : settings.BatchSize),
        UBatchSize = (uint)(requested == ModelKind.Vision ? settings.VisionMicroBatchSize : settings.MicroBatchSize)
    };

    public void UnloadVision() { lock (sync) { if (kind == ModelKind.Vision) Unload(); } }
    public void Stop() { lock (sync) { active = false; Unload(); } }
    public void ScheduleStop()
    {
        lock (sync)
        {
            active = false;
            idleTimer?.Dispose();
            idleTimer = new Timer(_ => { lock (sync) { if (!active) Unload(); } }, null,
                TimeSpan.FromMinutes(Math.Max(1, settings.IdleMinutes)), Timeout.InfiniteTimeSpan);
        }
    }
    private void Unload() { idleTimer?.Dispose(); idleTimer = null; weights?.Dispose(); weights = null; kind = null; }
    public void Dispose() { lock (sync) { if (disposed) return; disposed = true; Unload(); } }
}
