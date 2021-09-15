package com.nvidia.grcuda.runtime.computation.streamattach;

import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

import java.util.Optional;
import java.util.concurrent.Callable;

/**
 * GPUs with pre-Pascal architecture or older, with compute capability < 6.0,
 * require to exclusively associate a managed memory array to a single stream to provide
 * asynchronous host access to managed memory while a kernel is running.
 * This interface wraps and executes the array association function specified in {@link GrCUDAComputationalElement}.
 * The array association function will be done only if the available GPU has compute capability < 6.0;
 */
public interface StreamAttachArchitecturePolicy {
    void execute(Runnable runnable);

    Optional<CUDAStream> execute(Callable<Optional<CUDAStream>> callable);
}
