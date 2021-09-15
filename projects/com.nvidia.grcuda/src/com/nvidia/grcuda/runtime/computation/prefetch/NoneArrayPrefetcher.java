package com.nvidia.grcuda.runtime.computation.prefetch;

import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;

public class NoneArrayPrefetcher extends AbstractArrayPrefetcher {

    public NoneArrayPrefetcher(CUDARuntime runtime) {
        super(runtime);
    }

    /**
     * This array prefetcher doesn't do anything;
     * @param computation a computational element whose array inputs can be prefetched from host to GPU
     */
    @Override
    public void prefetchToGpu(GrCUDAComputationalElement computation) {
    }
}
