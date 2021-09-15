package com.nvidia.grcuda.runtime.computation.prefetch;

import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;

/**
 * Class that declares an interface to prefetch the data from CPU to GPU (and possibly viceversa).
 * Prefetching requires a GPU with architecture starting from Pascal, and is not required for functionality (it is just a performance optimization).
 */
public abstract class AbstractArrayPrefetcher {

    protected CUDARuntime runtime;

    public AbstractArrayPrefetcher(CUDARuntime runtime) {
        this.runtime = runtime;
    }

    /**
     * Prefetch the arrays of a {@link GrCUDAComputationalElement}. Prefetching is always done asynchronously.
     * @param computation a computational element whose array inputs can be prefetched from host to GPU
     */
    public abstract void prefetchToGpu(GrCUDAComputationalElement computation);

    public void prefetchToGpu(ExecutionDAG.DAGVertex vertex) {
        this.prefetchToGpu(vertex.getComputation());
    }
}
