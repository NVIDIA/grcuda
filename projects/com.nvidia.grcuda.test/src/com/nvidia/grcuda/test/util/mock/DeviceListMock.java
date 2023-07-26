package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.DeviceList;

public class DeviceListMock extends DeviceList {
    public DeviceListMock(int numDevices) {
        super(numDevices, null);
    }

    @Override
    public void initializeDeviceList(int numDevices, CUDARuntime runtime) {
        for (int deviceOrdinal = 0; deviceOrdinal < numDevices; deviceOrdinal++) {
            devices.set(deviceOrdinal, new DeviceMock(deviceOrdinal, null));
        }
    }
}
