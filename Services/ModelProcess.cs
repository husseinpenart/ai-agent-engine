using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using Microsoft.Extensions.Options;

namespace ai_agent.Services;

public sealed class ModelProcess(IOptions<ModelOptions> options, ILogger<ModelProcess> logger) : IDisposable
{
    private readonly ModelOptions settings = options.Value;
    private readonly HttpClient probe = new() { Timeout = TimeSpan.FromSeconds(2) };
    private readonly Lock sync = new();
    private Process? process;
    private Timer? idleTimer;
    private Uri? address;
    private bool active;

    public async Task<Uri> StartAsync(CancellationToken cancellationToken)
    {
        lock (sync)
        {
            active = true;
            idleTimer?.Dispose();
            idleTimer = null;
            if (process is null || process.HasExited)
            {
                process?.Dispose();
                var executable = Environment.ExpandEnvironmentVariables(settings.ExecutablePath);
                var model = Environment.ExpandEnvironmentVariables(settings.ModelPath);
                var backend = Environment.ExpandEnvironmentVariables(settings.BackendDirectory);
                if (!File.Exists(executable)) throw new InvalidOperationException("llama-server executable not found: " + executable);
                if (!File.Exists(model)) throw new InvalidOperationException("GGUF model file not found: " + model);
                if (!Directory.Exists(backend)) throw new InvalidOperationException("CUDA backend directory not found: " + backend);
                using var listener = new TcpListener(IPAddress.Loopback, 0);
                listener.Start();
                var port = ((IPEndPoint)listener.LocalEndpoint).Port;
                listener.Stop();
                address = new Uri($"http://127.0.0.1:{port}/");
                var info = new ProcessStartInfo(executable) { WorkingDirectory = backend, UseShellExecute = false, CreateNoWindow = true, RedirectStandardError = true, RedirectStandardOutput = true };
                info.Environment["PATH"] = backend + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH");
                foreach (var argument in new[] { "--model", model, "--host", "127.0.0.1", "--port", port.ToString(), "--ctx-size", settings.ContextSize.ToString(), "--n-gpu-layers", settings.GpuLayers.ToString(), "--threads", settings.Threads.ToString(), "--parallel", "1", "--jinja" }) info.ArgumentList.Add(argument);
                process = Process.Start(info) ?? throw new InvalidOperationException("Failed to start llama-server.");
                process.ErrorDataReceived += (_, line) => { if (line.Data is not null) logger.LogDebug("Model: {Line}", line.Data); };
                process.OutputDataReceived += (_, line) => { if (line.Data is not null) logger.LogDebug("Model: {Line}", line.Data); };
                process.BeginErrorReadLine();
                process.BeginOutputReadLine();
            }
        }
        try
        {
            for (var attempt = 0; attempt < 600; attempt++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (process!.HasExited) throw new InvalidOperationException("The model process exited during startup. Check the model file and CUDA runtime.");
                try { using var response = await probe.GetAsync(new Uri(address!, "health"), cancellationToken); if (response.IsSuccessStatusCode) return address!; }
                catch (HttpRequestException) { }
                catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested) { }
                await Task.Delay(500, cancellationToken);
            }
            throw new TimeoutException("The model did not load within five minutes.");
        }
        catch { Stop(); throw; }
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
        lock (sync) { if (!active) Stop(); }
    }

    private void Stop()
    {
        lock (sync)
        {
            idleTimer?.Dispose();
            idleTimer = null;
            if (process is not null)
            {
                try { if (!process.HasExited) process.Kill(entireProcessTree: true); }
                catch (InvalidOperationException) { }
                process.Dispose();
                process = null;
                address = null;
            }
        }
    }

    public void Dispose() { Stop(); probe.Dispose(); }
}
