package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

public class DeviceMock extends Device {

    public DeviceMock(int deviceId, CUDARuntime runtime) {
        super(deviceId, runtime);
    }

    /**
     * Create a fake CUDA stream on this device
     */
    @Override
    public CUDAStream createStream() {
        CUDAStream newStream = new CUDAStream(0, GrCUDAStreamManagerMock.numUserAllocatedStreams++, deviceId);
        this.getStreams().add(newStream);
        return newStream;
    }

    @Override
    public void cleanup() {
        this.freeStreams.clear();
        this.getStreams().clear();
    }

    @Override
    public String toString() {
        return "MockGPU(id=" + deviceId + ")";
    }
}

