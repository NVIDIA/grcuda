package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.array.DeviceArray;
import com.oracle.truffle.api.profiles.ValueProfile;

public class DeviceArrayReadExecution extends GrCUDAComputationalElement {

    private final DeviceArray array;
    private final long index;
    private final ValueProfile elementTypeProfile;

    public DeviceArrayReadExecution(DeviceArray array,
                                     long index,
                                     ValueProfile elementTypeProfile) {
        super(array.getGrCUDAExecutionContext(), new DeviceArrayExecutionInitializer(array));
        this.array = array;
        this.index = index;
        this.elementTypeProfile = elementTypeProfile;
    }

    @Override
    public Object execute() {
        Object result = array.readArrayElementImpl(index, elementTypeProfile);
        this.setComputationFinished();
        return result;
    }

    @Override
    public String toString() {
        return "DeviceArrayReadExecution(" +
                "array=" + array +
                ", index=" + index + ")";
    }
}
