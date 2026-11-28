namespace ai_agent.Services;

public sealed class ModelOptions
{
    public string ProcessStateDirectory { get; set; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "my-agent");

    public string ModelPath { get; set; } = @"E:\AI_Models\codingmodel\Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf";
    public string VisionModelPath { get; set; } = @"E:\AI_Models\codingmodel\Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf";
    public string VisionMmprojPath { get; set; } = @"E:\AI_Models\codingmodel\mmproj-F32.gguf";
    public int CpuMoeLayers { get; set; } = 0;
    public int ContextSize { get; set; } = 8192;
    public int MaxOutputTokens { get; set; } = 4096;
    public int VisionContextSize { get; set; } = 8192;
    public int VisionMaxOutputTokens { get; set; } = 2048;
    public int VisionGpuLayers { get; set; } = 99;
    public int RecentMessages { get; set; } = 20;
    public string GpuLayers { get; set; } = "auto";
    public string GpuDevice { get; set; } = "CUDA0";
    public string FlashAttention { get; set; } = "auto";
    public int BatchSize { get; set; } = 1024;
    public int MicroBatchSize { get; set; } = 256;
    public int VisionBatchSize { get; set; } = 512;
    public int VisionMicroBatchSize { get; set; } = 128;
    public int GpuMemoryReserveMiB { get; set; } = 1024;
    public int ToolResultMaxChars { get; set; } = 200000;
    public int Threads { get; set; } = 6;
    public int IdleMinutes { get; set; } = 15;
}

public sealed class InferenceGate
{
    public SemaphoreSlim Semaphore { get; } = new(1, 1);
}