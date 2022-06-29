package com.nvidia.grcuda.runtime;

public class CPUDevice extends AbstractDevice {
    public static final int CPU_DEVICE_ID = -1;

    public CPUDevice() {
        super(CPU_DEVICE_ID);
    }

    @Override
    public String toString() {
        return "CPU(id=" + deviceId + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CPUDevice that = (CPUDevice) o;
        return deviceId == that.deviceId;
    }
}
