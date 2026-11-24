namespace ai_agent.Services;

internal sealed class ModelProcessLease : IDisposable
{
    private readonly ManualResetEvent release;
    private readonly Thread owner;

    private ModelProcessLease(ManualResetEvent release, Thread owner)
    {
        this.release = release;
        this.owner = owner;
    }

    public static Task<ModelProcessLease> AcquireAsync(CancellationToken cancellationToken)
    {
        var completion = new TaskCompletionSource<ModelProcessLease>(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new ManualResetEvent(false);
        Thread? owner = null;
        owner = new Thread(() =>
        {
            var acquired = false;
            try
            {
                using var mutex = new Mutex(false, "my-agent-model-process-v1");
                try { acquired = WaitHandle.WaitAny([mutex, cancellationToken.WaitHandle]) == 0; }
                catch (AbandonedMutexException) { acquired = true; }
                if (!acquired) { completion.TrySetCanceled(cancellationToken); return; }
                completion.TrySetResult(new ModelProcessLease(release, owner!));
                release.WaitOne();
                mutex.ReleaseMutex();
            }
            catch (Exception exception) { completion.TrySetException(exception); }
            finally { if (!acquired) release.Dispose(); }
        }) { IsBackground = true, Name = "my-agent-model-lease" };
        owner.Start();
        return completion.Task;
    }

    public void Dispose()
    {
        release.Set();
        owner.Join();
        release.Dispose();
    }
}
