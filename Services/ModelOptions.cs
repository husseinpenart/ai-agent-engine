namespace ai_agent.Services;

public sealed class ModelOptions
{
    public string BaseUrl { get; set; } = "http://127.0.0.1:11434";
    public string Name { get; set; } = "local-qwen-coder";
    public int ContextSize { get; set; } = 4096;
    public int MaxOutputTokens { get; set; } = 1024;
    public int GpuLayers { get; set; } = 8;
    public int Threads { get; set; } = 4;
    public string KeepAlive { get; set; } = "5m";
}

public sealed class InferenceGate
{
    public SemaphoreSlim Semaphore { get; } = new(1, 1);
}
