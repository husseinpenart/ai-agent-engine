# Local agent backend

The Tauri frontend connects to the ASP.NET API, for example at `https://localhost:44382` when using the IIS Express profile. The backend starts a private `llama-server` process on an available loopback port on the first chat request. This process reads the configured GGUF directly; the Ollama service and model import are not required. The executable and CUDA libraries currently come from the installed Ollama directory and can be changed in `appsettings.json`.

Start the backend with the IIS Express profile in Visual Studio, or run `dotnet run --no-launch-profile` and configure the frontend to use `http://localhost:5000`. In development, Swagger UI is available under `/swagger`. `GET /api/health` reports API availability; it does not load the model. `POST /api/agent/chat` streams SSE events and can take time on its first request while the model loads.

`appsettings.json` configures model and executable paths, CUDA backend directory, context size, CPU threads, GPU layers, output length and idle timeout. Defaults are one inference at a time, four CPU threads, 4096 context tokens, eight GPU layers and unloading after five idle minutes. The 17.7 GB model does not fit in 8 GB VRAM and uses system RAM; actual memory use and speed depend on the system. Close other memory-heavy applications and monitor RAM and VRAM when first loading it. Stopping the backend normally also stops the model process.

The API is intended for local use and has no authentication; do not expose it to a network. The client executes tools and manages permissions; desktop conversation storage is managed by the frontend. The chat model is text-only.
