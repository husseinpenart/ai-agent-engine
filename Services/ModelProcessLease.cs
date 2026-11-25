namespace ai_agent.Services;

internal sealed class ModelProcessLease : IDisposable
{
    private readonly ManualResetEvent release;
    private readonly Thread owner;
    private bool disposed;

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
            Mutex? mutex = null;
            try
            {
                mutex = new Mutex(false, "my-agent-model-process-v1");
                try
                {
                    acquired = WaitHandle.WaitAny([mutex, cancellationToken.WaitHandle]) == 0;
                }
                catch (AbandonedMutexException)
                {
                    acquired = true;
                }

                if (!acquired)
                {
                    completion.TrySetCanceled(cancellationToken);
                    return;
                }

                var leaseInstance = new ModelProcessLease(release, owner!);
                if (!completion.TrySetResult(leaseInstance))
                {
                    release.Set();
                    return;
                }

                release.WaitOne();
            }
            catch (Exception exception)
            {
                completion.TrySetException(exception);
            }
            finally
            {
                if (acquired && mutex is not null)
                {
                    try { mutex.ReleaseMutex(); }
                    catch (ApplicationException) { }
                }
                mutex?.Dispose();
                if (!acquired)
                {
                    try { release.Dispose(); }
                    catch (ObjectDisposedException) { }
                }
            }
        })
        {
            IsBackground = true,
            Name = "my-agent-model-lease"
        };

        owner.Start();
        return completion.Task;
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        try { release.Set(); }
        catch (ObjectDisposedException) { }
        if (owner.IsAlive)
        {
            owner.Join(TimeSpan.FromSeconds(10));
        }
        try { release.Dispose(); }
        catch (ObjectDisposedException) { }
    }
}