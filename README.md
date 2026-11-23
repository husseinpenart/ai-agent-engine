# Local agent backend

ASP.NET Core exposes `GET /api/health` and `POST /api/agent/chat` on `http://127.0.0.1:5000`. Chat accepts the Tauri client's messages and tool definitions and streams `text`, `thinking`, `tool_call`, `error`, and `done` SSE events. The client runs tools and handles permission prompts; this backend does not execute tools or store conversation history. The current frontend stores its sessions in localStorage.

## Setup

1. Start Ollama locally. Import the GGUF once from this directory:

   ```powershell
   ollama create local-qwen-coder -f Modelfile
   ```

   Importing can require approximately another copy of the 17.7 GB model in Ollama's model storage. Ensure the disk has enough free space. Ollama must support this GGUF quantization and its chat template; compatibility with this particular file has not been verified.

2. Run the API:

   ```powershell
   dotnet run --no-launch-profile
   ```

The model name, server address, context window, GPU layers, CPU threads, output limit and idle unloading time are configurable in `appsettings.json`, or with standard .NET configuration environment variables such as `Model__GpuLayers`. Defaults allow one active inference, four CPU threads, 4096 context tokens, and eight GPU layers. The model is much larger than 8 GB VRAM, so inference uses system RAM as well and may be slow. These values are starting points, not memory guarantees; monitor RAM and VRAM on the first run and reduce GPU layers or context if necessary. Ollama must be running separately; this API does not launch or install it.

Only the listed local development and Tauri origins are allowed by CORS. The API is intended for local use and has no authentication; do not expose it to a network. Health reports API availability, not whether Ollama or the model is ready. The supplied GGUF path is local to this machine and can be changed in `Modelfile` before import.
