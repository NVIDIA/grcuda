package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.DefaultStream;

public class KernelConfigBuilder {

    private final Dim3 gridSize;
    private final Dim3 blockSize;
    private int dynamicSharedMemoryBytes = 0;
    private CUDAStream stream = new DefaultStream();
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
