package com.nvidia.grcuda.runtime.computation.prefetch;

import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

public class SyncArrayPrefetcher extends AbstractArrayPrefetcher {

    public SyncArrayPrefetcher(CUDARuntime runtime) {
        super(runtime);
    }

    /**
     * The synchronous array prefetcher schedules prefetching on the arrays used by the computation, and waits for their completion.
     * Only the arrays whose last operation has been a CPU access are prefetched, as the other are already up-to-date on GPU.
     * The prefetcher assumes that the GPU allows prefetching (architecture since Pascal) and the arrays are visible to the stream where they are prefetched.
     * Technically, we need prefetching only if the array has been modified by the CPU, and we could prefetch only the part that has been modified;
     * this simple prefetcher still prefetches everything though.
     * @param computation a computational element whose array inputs can be prefetched from host to GPU
     */
    @Override
    public void prefetchToGpu(GrCUDAComputationalElement computation) {
        for (ComputationArgumentWithValue a : computation.getArgumentList()) {
            if (a.getArgumentValue() instanceof AbstractArray) {
                AbstractArray array = (AbstractArray) a.getArgumentValue();
                // The array has been used by the CPU, so we should prefetch it;
                if (array.isLastComputationArrayAccess()) {
                    CUDAStream streamToPrefetch = computation.getStream();
                    runtime.cudaMemPrefetchAsync(array, streamToPrefetch);
                    runtime.cudaStreamSynchronize(streamToPrefetch);
                }
            }
        }
    }
}
