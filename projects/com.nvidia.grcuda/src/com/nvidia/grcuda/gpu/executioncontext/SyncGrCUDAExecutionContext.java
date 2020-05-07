package com.nvidia.grcuda.gpu.executioncontext;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyComputationBuilder;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

/**
 * Execute all computations synchronously, without computing dependencies or using streams;
 */
public class SyncGrCUDAExecutionContext extends AbstractGrCUDAExecutionContext {

    public SyncGrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, DependencyComputationBuilder dependencyBuilder) {
        super(context, env, dependencyBuilder);
    }

    public SyncGrCUDAExecutionContext(CUDARuntime cudaRuntime, DependencyComputationBuilder dependencyBuilder) {
        super(cudaRuntime, dependencyBuilder);
    }

    /**
     * Register this computation for future execution by the {@link SyncGrCUDAExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data dependencies;
     */
    @Override
    public Object registerExecution(GrCUDAComputationalElement computation) throws UnsupportedTypeException {

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
