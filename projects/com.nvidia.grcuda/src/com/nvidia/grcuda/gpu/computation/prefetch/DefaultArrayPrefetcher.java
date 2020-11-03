package com.nvidia.grcuda.gpu.computation.prefetch;

import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.ParameterWithValue;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.stream.CUDAStream;

public class DefaultArrayPrefetcher extends AbstractArrayPrefetcher {

    public DefaultArrayPrefetcher(CUDARuntime runtime) {
        super(runtime);
    }

    /**
     * The default array prefetcher schedules asynchronous prefetching on the arrays used by the computation.
     * Only the arrays whose last operation has been a CPU access are prefetched, as the other are already up-to-date on GPU.
     * The prefetcher assumes that the GPU allows prefetching (architecture since Pascal) and the arrays are visible to the stream where they are prefetched.
     * Technically, we need prefetching only if the array has been modified by the CPU, and we could prefetch only the part that has been modified;
     * this simple prefetcher still prefetches everything though.
     * @param computation a computational element whose array inputs can be prefetched from host to GPU
     */
    @Override
    public void prefetchToGpu(GrCUDAComputationalElement computation) {
        for (ParameterWithValue a : computation.getArgumentList()) {
            if (a.getArgumentValue() instanceof AbstractArray) {
                AbstractArray array = (AbstractArray) a.getArgumentValue();
                // The array has been used by the CPU, so we should prefetch it;
                if (array.isLastComputationArrayAccess()) {
                    CUDAStream streamToPrefetch = computation.getStream();
                    runtime.cudaMemPrefetchAsync(array, streamToPrefetch);
                }
            }
        }
    }
}
