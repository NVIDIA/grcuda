package com.nvidia.grcuda;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

public class GrCUDAThreadManager {

    private final ExecutorService threadPool;
    private final GrCUDAContext context;
    protected final List<Thread> toJoin;

    public GrCUDAThreadManager(GrCUDAContext context) {
        this(context, context.getNumberOfThreads());
    }

    public GrCUDAThreadManager(GrCUDAContext context, int numberOfThreads) {
        this.toJoin = new LinkedList<>();
        this.context = context;
        this.threadPool = Executors.newFixedThreadPool(numberOfThreads, this::createJavaThread);
    }

    protected Thread createJavaThread(Runnable runnable) {
        Thread thread = context.getEnv().createThread(runnable);
        toJoin.add(thread);
        System.out.println("-- created thread " + thread);
        return thread;
    }

    public void submitRunnable(Runnable task) {
        threadPool.submit(task);
    }

    public <T> Future<T> submitCallable(Callable<T> task) {
        return threadPool.submit(task);
    }

    public <T> void submitTask(FutureTask<T> task) {
        threadPool.submit(task);
    }

    public <T> List<T> getResults(Collection<Future<T>> futures) {
        List<T> results = new LinkedList<>();
        futures.forEach(f -> {
            try {
                results.add(f.get());
            } catch (InterruptedException | ExecutionException e) {
                System.out.println("Failed to get result of future, exception: " + e);
                e.printStackTrace();
            }
        });
        return results;
    }

    public ExecutorService getThreadPool() {
        return threadPool;
    }

    public void finalizeManager() {
        if (threadPool == null)
            return;
//        System.out.println("closing GrCUDA thread manager...");
        threadPool.shutdown();
        try {
            if (!threadPool.awaitTermination(60, TimeUnit.SECONDS)) {
                threadPool.shutdownNow();
            }
            threadPool.awaitTermination(60, TimeUnit.SECONDS);

            for (Thread t : toJoin) {
                t.join();
            }

        } catch (InterruptedException ie) {
            threadPool.shutdownNow();
            Thread.currentThread().interrupt();
        }
//        System.out.println("closed GrCUDA thread manager");
    }
}