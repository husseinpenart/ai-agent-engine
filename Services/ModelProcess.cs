using System.Diagnostics;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using Microsoft.Extensions.Options;

namespace ai_agent.Services;

public enum ModelKind { Coder, Vision }

public sealed class ModelProcess(IOptions<ModelOptions> options, ILogger<ModelProcess> logger) : IDisposable
{
    private readonly ModelOptions settings = options.Value;
    private readonly HttpClient probe = new() { Timeout = TimeSpan.FromSeconds(2) };
    private readonly Lock sync = new();
    private readonly SemaphoreSlim lifecycle = new(1, 1);
    private readonly CancellationTokenSource shutdown = new();
    private Process? process;
    private ModelProcessJob? job;
    private ModelProcessLease? lease;
    private Timer? idleTimer;
    private Uri? address;
    private bool active;
    private ModelKind? loadedKind;
    private readonly ConcurrentQueue<string> diagnostics = new();
    private bool disposed;
    private string OwnershipPath => Path.Combine(Environment.ExpandEnvironmentVariables(settings.ProcessStateDirectory), "model-process.json");
    private sealed record OwnedProcess(int Pid, long StartTimeUtcTicks, string Executable, int Port);

    public async Task<Uri> StartAsync(ModelKind kind, CancellationToken cancellationToken)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, shutdown.Token);
        await lifecycle.WaitAsync(linked.Token);
        try
        {
            linked.Token.ThrowIfCancellationRequested();
            if (lease is null)
            {
                lease = await ModelProcessLease.AcquireAsync(linked.Token);
                linked.Token.ThrowIfCancellationRequested();
                CleanupOwnedProcess();
            }
            lock (sync)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                active = true;
                idleTimer?.Dispose();
                idleTimer = null;
                if (process is not null && (loadedKind != kind || process.HasExited)) StopCore(false);
                if (process is null || process.HasExited)
                {
                    process?.Dispose();
                    var executable = Environment.ExpandEnvironmentVariables(settings.ExecutablePath);
                    var model = Environment.ExpandEnvironmentVariables(kind == ModelKind.Vision ? settings.VisionModelPath : settings.ModelPath);
                    var projector = Environment.ExpandEnvironmentVariables(settings.VisionMmprojPath);
                    var backend = Environment.ExpandEnvironmentVariables(settings.BackendDirectory);
                    if (!File.Exists(executable)) throw new InvalidOperationException("llama-server executable not found: " + executable);
                    if (!File.Exists(model)) throw new InvalidOperationException("GGUF model file not found: " + model);
                    if (kind == ModelKind.Vision && !File.Exists(projector)) throw new InvalidOperationException("Vision projector file not found: " + projector);
                    if (!Directory.Exists(backend)) throw new InvalidOperationException("CUDA backend directory not found: " + backend);
                    using var listener = new TcpListener(IPAddress.Loopback, 0);
                    listener.Start();
                    var port = ((IPEndPoint)listener.LocalEndpoint).Port;
                    listener.Stop();
                    address = new Uri($"http://127.0.0.1:{port}/");
                    var info = new ProcessStartInfo(executable) { WorkingDirectory = backend, UseShellExecute = false, CreateNoWindow = true, RedirectStandardError = true, RedirectStandardOutput = true };
                    info.Environment["PATH"] = backend + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH");
                    var gpuLayers = kind == ModelKind.Vision ? settings.VisionGpuLayers.ToString() : settings.GpuLayers;
                    var batch = kind == ModelKind.Vision ? settings.VisionBatchSize : settings.BatchSize;
                    var microBatch = kind == ModelKind.Vision ? settings.VisionMicroBatchSize : settings.MicroBatchSize;
                    if (batch < 1 || microBatch < 1 || microBatch > batch || settings.GpuMemoryReserveMiB < 0 || string.IsNullOrWhiteSpace(settings.GpuDevice) || settings.FlashAttention is not ("on" or "off" or "auto")) throw new InvalidOperationException("Invalid model batch, GPU device, reserve, or Flash Attention settings.");
                    foreach (var argument in new[] { "--model", model, "--host", "127.0.0.1", "--port", port.ToString(), "--ctx-size", (kind == ModelKind.Vision ? settings.VisionContextSize : settings.ContextSize).ToString(), "--device", settings.GpuDevice, "--n-gpu-layers", gpuLayers, "--threads", settings.Threads.ToString(), "--parallel", "1", "--jinja", "--batch-size", batch.ToString(), "--ubatch-size", microBatch.ToString(), "--flash-attn", settings.FlashAttention, "--fit", "on", "--fit-target", settings.GpuMemoryReserveMiB.ToString() }) info.ArgumentList.Add(argument);
                    if (kind == ModelKind.Vision) foreach (var argument in new[] { "--mmproj", projector, "--mmproj-offload", "--mmproj-device", settings.GpuDevice, "--image-max-tokens", "1024" }) info.ArgumentList.Add(argument);
                    diagnostics.Clear();
                    process = Process.Start(info) ?? throw new InvalidOperationException("Failed to start llama-server.");
                    logger.LogInformation("Started local {ModelKind} model: PID {Pid}, port {Port}", kind, process.Id, port);
                    try
                    {
                        job = ModelProcessJob.Attach(process);
                        SaveOwnership(new OwnedProcess(process.Id, process.StartTime.ToUniversalTime().Ticks, Path.GetFullPath(executable), port));
                    }
                    catch { StopCore(false); throw; }
                    loadedKind = kind;
                    process.ErrorDataReceived += (sender, line) =>
                    {
                        if (line.Data is not { } data) return;
                        logger.LogDebug("Model: {Line}", data);
                        if (data.Contains("offloaded", StringComparison.OrdinalIgnoreCase) || data.Contains("using device", StringComparison.OrdinalIgnoreCase)) logger.LogInformation("Model: {Line}", data);
                        if (data.Contains("error", StringComparison.OrdinalIgnoreCase) || data.Contains("failed", StringComparison.OrdinalIgnoreCase) || data.Contains("out of memory", StringComparison.OrdinalIgnoreCase))
                        {
                            diagnostics.Enqueue(data.Length > 500 ? data[..500] : data);
                            while (diagnostics.Count > 6) diagnostics.TryDequeue(out _);
                        }
                    };
                    process.OutputDataReceived += (_, line) => { if (line.Data is not null) logger.LogDebug("Model: {Line}", line.Data); };
                    process.BeginErrorReadLine();
                    process.BeginOutputReadLine();
                }
            }
            for (var attempt = 0; attempt < 600; attempt++)
            {
                linked.Token.ThrowIfCancellationRequested();
                if (process!.HasExited)
                {
                    process.WaitForExit();
                    throw new InvalidOperationException($"The {kind} model process exited during startup (code {process.ExitCode}). " + string.Join(" | ", diagnostics));
                }
                try { using var response = await probe.GetAsync(new Uri(address!, "health"), linked.Token); if (response.IsSuccessStatusCode) return address!; }
                catch (HttpRequestException) { }
                catch (TaskCanceledException) when (!linked.IsCancellationRequested) { }
                await Task.Delay(500, linked.Token);
            }
            throw new TimeoutException("The model did not load within five minutes.");
        }
        catch { lock (sync) StopCore(true); throw; }
        finally { lifecycle.Release(); }
    }

    public void ScheduleStop()
    {
        lock (sync)
        {
            active = false;
            idleTimer?.Dispose();
            idleTimer = new Timer(_ => StopIfIdle(), null, TimeSpan.FromMinutes(Math.Max(1, settings.IdleMinutes)), Timeout.InfiniteTimeSpan);
        }
    }

    private void StopIfIdle()
    {
        lifecycle.Wait();
        try { lock (sync) { if (!active && !disposed) StopCore(true); } }
        catch (Exception exception) { logger.LogError(exception, "Unable to stop the idle model process"); }
        finally { lifecycle.Release(); }
    }

    public void UnloadVision()
    {
        lifecycle.Wait();
        try { lock (sync) { if (loadedKind == ModelKind.Vision) StopCore(false); } }
        finally { lifecycle.Release(); }
    }

    public string? ExitedProcessDetail()
    {
        lock (sync)
        {
            return process is not null && process.HasExited ? $"Local {loadedKind} model process exited with code {process.ExitCode}." : null;
        }
    }

    public void Stop()
    {
        lifecycle.Wait();
        try { lock (sync) StopCore(true); }
        finally { lifecycle.Release(); }
    }

    private void StopCore(bool releaseLease)
    {
        lock (sync)
        {
            idleTimer?.Dispose();
            idleTimer = null;
            if (process is not null)
            {
                var pid = process.Id;
                var port = address?.Port;
                try
                {
                    if (!process.HasExited)
                    {
                        logger.LogInformation("Stopping local model: PID {Pid}, port {Port}", pid, port);
                        process.Kill(entireProcessTree: true);
                        if (!process.WaitForExit(15000)) throw new TimeoutException("Model process did not exit; refusing to load another model.");
                        logger.LogInformation("Stopped local model: PID {Pid}, port {Port}", pid, port);
                    }
                }
                catch (InvalidOperationException) when (process.HasExited) { }
                job?.Dispose();
                job = null;
                process.Dispose();
                process = null;
                address = null;
                loadedKind = null;
                DeleteOwnership();
            }
            if (releaseLease) { lease?.Dispose(); lease = null; }
        }
    }

    private void SaveOwnership(OwnedProcess owned)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(OwnershipPath)!);
        var temporary = OwnershipPath + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try { File.WriteAllText(temporary, JsonSerializer.Serialize(owned)); File.Move(temporary, OwnershipPath, true); }
        finally { File.Delete(temporary); }
    }

    private void DeleteOwnership()
    {
        try { File.Delete(OwnershipPath); }
        catch (DirectoryNotFoundException) { }
    }

    private void CleanupOwnedProcess()
    {
        string contents;
        try { contents = File.ReadAllText(OwnershipPath); }
        catch (FileNotFoundException) { return; }
        catch (DirectoryNotFoundException) { return; }
        var owned = JsonSerializer.Deserialize<OwnedProcess>(contents) ?? throw new InvalidOperationException("Invalid model process ownership record.");
        Process candidate;
        try { candidate = Process.GetProcessById(owned.Pid); }
        catch (ArgumentException) { DeleteOwnership(); return; }
        using (candidate)
        {
            if (candidate.HasExited || candidate.StartTime.ToUniversalTime().Ticks != owned.StartTimeUtcTicks) { DeleteOwnership(); return; }
            var executable = candidate.MainModule?.FileName ?? throw new InvalidOperationException("Cannot verify the previous model process executable.");
            if (!string.Equals(Path.GetFullPath(executable), owned.Executable, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal)) { DeleteOwnership(); return; }
            logger.LogWarning("Stopping previously owned model process: PID {Pid}, port {Port}", owned.Pid, owned.Port);
            try { candidate.Kill(entireProcessTree: true); }
            catch (InvalidOperationException) when (candidate.HasExited) { }
            if (!candidate.WaitForExit(15000)) throw new TimeoutException("Previous model process did not exit; refusing to start another model.");
            logger.LogInformation("Stopped previously owned model process: PID {Pid}, port {Port}", owned.Pid, owned.Port);
        }
        DeleteOwnership();
    }

    public void Dispose()
    {
        shutdown.Cancel();
        lifecycle.Wait();
        try { lock (sync) { if (!disposed) { disposed = true; StopCore(true); probe.Dispose(); } } }
        finally { lifecycle.Release(); }
    }
}
