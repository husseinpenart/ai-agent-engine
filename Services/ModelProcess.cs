using System.Text;
using LLama;
using LLama.Abstractions;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using Microsoft.Extensions.Options;

namespace ai_agent.Services;

public enum ModelKind { Coder, Vision }

public sealed class ModelProcess : IDisposable
{
    private readonly ModelOptions _settings;
    private readonly ILogger<ModelProcess> _logger;
    private readonly SemaphoreSlim _lifecycle = new(1, 1);
    private readonly CancellationTokenSource _shutdown = new();

    private LLamaWeights? _weights;
    private ModelParams? _params;
    private StatelessExecutor? _executor;
    private ModelKind? _loadedKind;
    private Timer? _idleTimer;
    private volatile bool _active;
    private volatile bool _disposed;

    public ModelKind? LoadedKind => _loadedKind;

    public ModelProcess(IOptions<ModelOptions> options, ILogger<ModelProcess> logger)
    {
        _settings = options.Value;
        _logger = logger;

        NativeLibraryConfig.All
            .WithCuda(true)
            .WithVulkan(true)
            .WithAutoFallback(true);
    }

    public async Task EnsureLoadedAsync(ModelKind kind, CancellationToken cancellationToken)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _shutdown.Token);
        await _lifecycle.WaitAsync(linked.Token);
        try
        {
            await EnsureLoadedCoreAsync(kind, linked.Token);
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    private async Task EnsureLoadedCoreAsync(ModelKind kind, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _active = true;
        _idleTimer?.Dispose();
        _idleTimer = null;

        if (_executor is not null && _loadedKind == kind)
            return;

        UnloadCore();

        var modelPath = Environment.ExpandEnvironmentVariables(
            kind == ModelKind.Vision ? _settings.VisionModelPath : _settings.ModelPath);

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"GGUF model file not found: {modelPath}", modelPath);

        if (kind == ModelKind.Vision)
        {
            var mmprojPath = Environment.ExpandEnvironmentVariables(_settings.VisionMmprojPath);
            if (!File.Exists(mmprojPath))
                throw new FileNotFoundException($"Vision mmproj file not found: {mmprojPath}", mmprojPath);
        }

        var parameters = BuildParams(kind, modelPath);

        await Task.Run(() =>
        {
            var weights = LLamaWeights.LoadFromFile(parameters);
            _weights = weights;
            _params = parameters;
            _executor = new StatelessExecutor(weights, parameters);
            _loadedKind = kind;
        }, cancellationToken);

        _logger.LogInformation(
            "Loaded {Kind} GGUF model: {Path} (GpuLayers={GpuLayers}, Context={Ctx}, Batch={Batch}/{UBatch}, CpuMoeLayers={Moe})",
            kind, modelPath, parameters.GpuLayerCount, parameters.ContextSize, parameters.BatchSize, parameters.UBatchSize,
            kind == ModelKind.Coder ? _settings.CpuMoeLayers : 0);
    }

    private ModelParams BuildParams(ModelKind kind, string modelPath)
    {
        var vision = kind == ModelKind.Vision;
        var threads = Math.Clamp(_settings.Threads, 1, Math.Max(1, Environment.ProcessorCount));

        var p = new ModelParams(modelPath)
        {
            ContextSize = (uint)Math.Max(1024, vision ? _settings.VisionContextSize : _settings.ContextSize),
            GpuLayerCount = ParseGpuLayers(vision ? _settings.VisionGpuLayers.ToString() : _settings.GpuLayers),
            Threads = threads,
            BatchThreads = threads,
            BatchSize = (uint)Math.Max(64, vision ? _settings.VisionBatchSize : _settings.BatchSize),
            UBatchSize = (uint)Math.Max(32, vision ? _settings.VisionMicroBatchSize : _settings.MicroBatchSize),
            FlashAttention = true
        };

        // MoE trick (same as llama.cpp --n-cpu-moe): keep attention on the GPU, put the expert
        // weights of the first N layers in system RAM. Much faster than letting VRAM overflow.
        // If TensorBufferOverride does not compile in your LLamaSharp version, delete this block.
        if (!vision && _settings.CpuMoeLayers > 0)
        {
            var layers = string.Join("|", Enumerable.Range(0, _settings.CpuMoeLayers));
            p.TensorBufferOverrides.Add(new TensorBufferOverride($@"blk\.({layers})\.ffn_.*_exps\.", "CPU"));
        }

        return p;
    }

    private static InferenceParams BuildInference(int maxTokens, float temperature, IReadOnlyList<string> stopSequences) => new()
    {
        MaxTokens = Math.Max(64, maxTokens),
        AntiPrompts = stopSequences.ToList(),
        SamplingPipeline = new DefaultSamplingPipeline
        {
            Temperature = temperature,
            TopK = 20,
            TopP = 0.8f,
            RepeatPenalty = 1.0f // repeat penalties corrupt code (brackets, indentation, identifiers)
        }
    };

    public async Task<string> InferTextAsync(
        ModelKind kind,
        string prompt,
        int maxTokens,
        float temperature,
        IReadOnlyList<string> stopSequences,
        CancellationToken cancellationToken)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _shutdown.Token);
        await _lifecycle.WaitAsync(linked.Token);
        try
        {
            await EnsureLoadedCoreAsync(kind, linked.Token);

            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_executor is null)
                throw new InvalidOperationException("Model executor is not initialized.");

            var sb = new StringBuilder();
            await foreach (var token in _executor.InferAsync(prompt, BuildInference(maxTokens, temperature, stopSequences), linked.Token))
            {
                sb.Append(token);
            }

            return sb.ToString().Trim();
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    /// <summary>
    /// Real image understanding through the multimodal projector (mtmd).
    /// The prompt must contain one "&lt;__media__&gt;" marker per image, in the same order as the images.
    /// </summary>
    public async Task<string> InferVisionAsync(
        string prompt,
        IReadOnlyList<byte[]> imageBytesList,
        int maxTokens,
        float temperature,
        IReadOnlyList<string> stopSequences,
        CancellationToken cancellationToken)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _shutdown.Token);
        await _lifecycle.WaitAsync(linked.Token);
        try
        {
            await EnsureLoadedCoreAsync(ModelKind.Vision, linked.Token);

            var mmprojPath = Environment.ExpandEnvironmentVariables(_settings.VisionMmprojPath);
            if (!File.Exists(mmprojPath))
                throw new FileNotFoundException($"Vision mmproj file not found: {mmprojPath}", mmprojPath);

            var weights = _weights ?? throw new InvalidOperationException("Vision model is not loaded.");
            var parameters = _params ?? throw new InvalidOperationException("Vision parameters are missing.");

            using var context = weights.CreateContext(parameters);

            var mtmdParams = MtmdContextParams.Default();
            mtmdParams.UseGpu = true;
            mtmdParams.NThreads = _settings.Threads;
            mtmdParams.ImageMaxTokens = 1024;

            using var mtmd = MtmdWeights.LoadFromFile(mmprojPath, weights, mtmdParams);
            if (!mtmd.SupportsVision)
                throw new InvalidOperationException("The projector does not support image input.");

            foreach (var bytes in imageBytesList)
                mtmd.LoadMedia(bytes);

            var executor = new InteractiveExecutor(context, mtmd);
            var sb = new StringBuilder();
            await foreach (var token in executor.InferAsync(prompt, BuildInference(maxTokens, temperature, stopSequences), linked.Token))
            {
                sb.Append(token);
            }

            return sb.ToString().Trim();
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    public void ScheduleStop()
    {
        if (_disposed) return;
        _active = false;
        _idleTimer?.Dispose();
        _idleTimer = new Timer(_ =>
        {
            try
            {
                if (_active || _disposed) return;
                if (!_lifecycle.Wait(0)) return; // somebody is using the model right now
                try
                {
                    if (!_active && !_disposed) UnloadCore();
                }
                finally
                {
                    _lifecycle.Release();
                }
            }
            catch (ObjectDisposedException)
            {
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to unload idle model");
            }
        }, null, TimeSpan.FromMinutes(Math.Max(1, _settings.IdleMinutes)), Timeout.InfiniteTimeSpan);
    }

    public void UnloadVision()
    {
        if (_loadedKind != ModelKind.Vision) return;
        _lifecycle.Wait();
        try
        {
            if (_loadedKind == ModelKind.Vision)
                UnloadCore();
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    private void UnloadCore()
    {
        _executor = null;
        _params = null;
        _weights?.Dispose();
        _weights = null;
        _loadedKind = null;
    }

    private static int ParseGpuLayers(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw) || raw.Equals("auto", StringComparison.OrdinalIgnoreCase))
            return 99;
        return int.TryParse(raw, out var n) ? Math.Max(0, n) : 35;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _shutdown.Cancel();
        _idleTimer?.Dispose();
        UnloadCore();
        _lifecycle.Dispose();
        _shutdown.Dispose();
    }
}