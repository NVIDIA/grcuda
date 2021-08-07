package com.nvidia.grcuda.gpu.executioncontext;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.computation.prefetch.PrefetcherEnum;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

/**
 * Execute all computations synchronously, without computing dependencies or using streams;
 */
public class SyncGrCUDAExecutionContext extends AbstractGrCUDAExecutionContext {

    public SyncGrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, DependencyPolicyEnum dependencyPolicy, PrefetcherEnum inputPrefetch) {
        super(context, env, dependencyPolicy, inputPrefetch, ExecutionPolicyEnum.SYNC);
    }

    public SyncGrCUDAExecutionContext(CUDARuntime cudaRuntime, DependencyPolicyEnum dependencyPolicy) {
        super(cudaRuntime, dependencyPolicy, PrefetcherEnum.NONE, ExecutionPolicyEnum.SYNC);
    }

    public SyncGrCUDAExecutionContext(CUDARuntime cudaRuntime, DependencyPolicyEnum dependencyPolicy, PrefetcherEnum inputPrefetch) {
        super(cudaRuntime, dependencyPolicy, inputPrefetch, ExecutionPolicyEnum.SYNC);
    }

    /**
     * Register this computation for future execution by the {@link SyncGrCUDAExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data dependencies;
     */
    @Override
    public Object registerExecution(GrCUDAComputationalElement computation) throws UnsupportedTypeException {

        // Prefetching;
        arrayPrefetcher.prefetchToGpu(computation);

        // Book-keeping;
        computation.setComputationStarted();
        computation.updateIsComputationArrayAccess();

        // Start the computation immediately;
        Object result = computation.execute();

        // Wait for the computation to end;
        cudaRuntime.cudaDeviceSynchronize();

        return result;
    }

    /**
     * All computations are synchronous, and atomic;
     * @return false
     */
    @Override
    public boolean isAnyComputationActive() {
        return false;
    }
}
