namespace ai_agent.Services;

public sealed class ModelOptions
{
    public string ExecutablePath { get; set; } = @"%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-server.exe";
    public string BackendDirectory { get; set; } = @"%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v12";
    public string ModelPath { get; set; } = @"E:\AI_Models\codingmodel\Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf";
    public string VisionModelPath { get; set; } = @"E:\AI_Models\codingmodel\Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf";
    public string VisionMmprojPath { get; set; } = @"E:\AI_Models\codingmodel\mmproj-F32.gguf";
    public int ContextSize { get; set; } = 8192;
    public int MaxOutputTokens { get; set; } = 4096;
    public int VisionContextSize { get; set; } = 4096;
    public int VisionMaxOutputTokens { get; set; } = 1024;
    public int VisionGpuLayers { get; set; } = 8;
    public int RecentMessages { get; set; } = 14;
    public int GpuLayers { get; set; } = 8;
    public int Threads { get; set; } = 4;
    public int IdleMinutes { get; set; } = 5;
}

public sealed class InferenceGate
{
    public SemaphoreSlim Semaphore { get; } = new(1, 1);
}
