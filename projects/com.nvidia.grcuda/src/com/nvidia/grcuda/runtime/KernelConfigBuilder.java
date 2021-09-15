package com.nvidia.grcuda.runtime;

import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.stream.DefaultStream;

public class KernelConfigBuilder {

    private final Dim3 gridSize;
    private final Dim3 blockSize;
    private int dynamicSharedMemoryBytes = 0;
    private CUDAStream stream = DefaultStream.get();
    private boolean useCustomStream = false;

    KernelConfigBuilder(Dim3 gridSize, Dim3 blockSize) {
        this.gridSize = gridSize;
        this.blockSize = blockSize;
    }

    public static KernelConfigBuilder newBuilder(Dim3 gridSize, Dim3 blockSize) {
        return new KernelConfigBuilder(gridSize, blockSize);
    }

    public KernelConfigBuilder dynamicSharedMemoryBytes(int bytes) {
        this.dynamicSharedMemoryBytes = bytes;
        return this;
    }

    public KernelConfigBuilder stream(CUDAStream stream) {
        this.stream = stream;
        this.useCustomStream = true;
        return this;
    }

    public KernelConfig build() {
        return new KernelConfig(gridSize, blockSize, dynamicSharedMemoryBytes, stream, useCustomStream);
    }
}
