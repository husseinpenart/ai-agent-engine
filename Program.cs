using ai_agent.Services;
using Microsoft.Extensions.FileProviders;

var builder = WebApplication.CreateBuilder(args);
builder.WebHost.UseUrls("http://127.0.0.1:5000");
builder.Services.Configure<ModelOptions>(builder.Configuration.GetSection("Model"));
builder.Services.AddHttpClient<LocalModelClient>(client => client.Timeout = Timeout.InfiniteTimeSpan);
builder.Services.AddHttpClient<VisionClient>(client => client.Timeout = Timeout.InfiniteTimeSpan);
builder.Services.AddSingleton<ConversationContext>();
builder.Services.AddSingleton<ModelProcess>();
builder.Services.AddSingleton<InferenceGate>();
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddCors(options => options.AddDefaultPolicy(policy => policy
    .WithOrigins("http://localhost:1420", "http://127.0.0.1:1420", "tauri://localhost", "http://tauri.localhost")
    .AllowAnyHeader().AllowAnyMethod()));
var app = builder.Build();
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}
app.UseCors();
app.Use(async (context, next) =>
{
    try { await next(); }
    catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
    {
        app.Logger.LogWarning(exception, "File access failed while handling {Path}", context.Request.Path);
        if (context.Response.HasStarted) context.Abort();
        else
        {
            context.Response.Clear();
            context.Response.StatusCode = StatusCodes.Status503ServiceUnavailable;
            await context.Response.WriteAsync("File temporarily unavailable.", context.RequestAborted);
        }
    }
});
if (Directory.Exists(app.Environment.WebRootPath)) app.UseStaticFiles();
var previewDirectory = builder.Configuration["StaticFiles:PreviewDirectory"];
if (!string.IsNullOrWhiteSpace(previewDirectory))
{
    try
    {
        var path = Path.GetFullPath(previewDirectory, app.Environment.ContentRootPath);
        if (Directory.Exists(path)) app.UseStaticFiles(new StaticFileOptions { FileProvider = new PhysicalFileProvider(path), RequestPath = "/preview" });
        else app.Logger.LogWarning("Preview directory does not exist: {Path}", path);
    }
    catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
    {
        app.Logger.LogWarning(exception, "Unable to configure the preview directory");
    }
}
app.MapControllers();
app.Run();
