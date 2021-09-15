package com.nvidia.grcuda.runtime.computation.streamattach;

import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.stream.DefaultStream;

import java.util.Optional;
import java.util.concurrent.Callable;

/**
 * GPUs with pre-Pascal architecture or older, with compute capability < 6.0,
 * require to exclusively associate a managed memory array to a single stream to provide
 * asynchronous host access to managed memory while a kernel is running.
 * This interface wraps and executes the array association function specified in {@link GrCUDAComputationalElement}
 */
public class PrePascalStreamAttachPolicy implements StreamAttachArchitecturePolicy {

    @Override
    public void execute(Runnable runnable) {
        runnable.run();
    }

    @Override
    public Optional<CUDAStream> execute(Callable<Optional<CUDAStream>> callable) {
        try {
            return callable.call();
        } catch(Exception e) {
            System.out.println("WARNING: failed to compute stream dependency, returning default stream");
            return Optional.of(DefaultStream.get());
        }
    }
}
