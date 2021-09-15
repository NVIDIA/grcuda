package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.GrCUDAThreadManager;

public class GrCUDAThreadManagerMock extends GrCUDAThreadManager {
    public GrCUDAThreadManagerMock(int numberOfThreads) {
        super(null, numberOfThreads);
    }

    @Override
    protected Thread createJavaThread(Runnable runnable) {
        Thread thread = new Thread(runnable);
        toJoin.add(thread);
        return thread;
    }
}
