namespace ai_agent.Services;

public sealed class ModelOptions
{
    public string ExecutablePath { get; set; } = @"%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-server.exe";
    public string BackendDirectory { get; set; } = @"%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v12";
    public string ModelPath { get; set; } = @"E:\AI_Models\codingmodel\Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf";
    public int ContextSize { get; set; } = 4096;
    public int MaxOutputTokens { get; set; } = 1024;
    public int GpuLayers { get; set; } = 8;
    public int Threads { get; set; } = 4;
    public int IdleMinutes { get; set; } = 5;
}

public sealed class InferenceGate
{
    public SemaphoreSlim Semaphore { get; } = new(1, 1);
}
