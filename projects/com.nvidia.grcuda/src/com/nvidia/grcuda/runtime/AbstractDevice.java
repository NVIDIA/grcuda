package com.nvidia.grcuda.runtime;

import java.util.Objects;

/**
 * Abstract device representation, used to distinguish between CPU and GPU devices inside the GrCUDA scheduler.
 */
public abstract class AbstractDevice {
    protected final int deviceId;

    public AbstractDevice(int deviceId) {
        this.deviceId = deviceId;
    }

    public int getDeviceId() {
        return deviceId;
    }

    @Override
    public String toString() {
        return "Device(id=" + deviceId + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AbstractDevice that = (AbstractDevice) o;
        return deviceId == that.deviceId;
    }

    @Override
    public int hashCode() {
        return Objects.hash(deviceId);
    }
}
