package com.nvidia.grcuda.gpu.stream;

import com.nvidia.grcuda.GrCUDAThreadManager;
import com.nvidia.grcuda.gpu.CUDARuntime;

import java.util.ArrayList;
import java.util.List;

public class GrCUDAStreamManager {

    /**
     * List of {@link CUDAStream} that have been currently allocated;
     */
    List<CUDAStream> streams = new ArrayList<>();
    /**
     * Reference to the CUDA runtime that manages the streams;
     */
    CUDARuntime runtime;

    public GrCUDAStreamManager(CUDARuntime runtime) {
        this.runtime = runtime;
    }

    /**
     * Create a new {@link CUDAStream} and add it to this manager, then return it;
     */
    public CUDAStream createStream() {
        CUDAStream newStream = runtime.cudaStreamCreate();
        streams.add(newStream);
        return newStream;
    }

    /**
     * Obtain the number of streams managed by this manager;
     */
    public int getNumberOfStreams() {
        return streams.size();
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        streams.forEach(runtime::cudaStreamDestroy);
    }
}
