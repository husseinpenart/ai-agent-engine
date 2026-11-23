using ai_agent.Services;

var builder = WebApplication.CreateBuilder(args);
builder.WebHost.UseUrls("http://127.0.0.1:5000");
builder.Services.Configure<ModelOptions>(builder.Configuration.GetSection("Model"));
builder.Services.AddHttpClient<LocalModelClient>(client => client.Timeout = Timeout.InfiniteTimeSpan);
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
app.MapControllers();
app.Run();
