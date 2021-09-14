package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.array.MultiDimDeviceArrayView;
import com.oracle.truffle.api.profiles.ValueProfile;

public class MultiDimDeviceArrayViewReadExecution extends ArrayAccessExecution<MultiDimDeviceArrayView> {

    private final long index;
    private final ValueProfile elementTypeProfile;

    public MultiDimDeviceArrayViewReadExecution(MultiDimDeviceArrayView array,
                                                long index,
                                                ValueProfile elementTypeProfile) {
        super(array.getGrCUDAExecutionContext(), new ArrayAccessExecutionInitializer<>(array.getMdDeviceArray(), true), array);
        this.index = index;
        this.elementTypeProfile = elementTypeProfile;
    }

    @Override
    public Object execute() {
        Object result = array.readNativeView(index, elementTypeProfile);
        this.setComputationFinished();
        return result;
    }

    @Override
    public String toString() {
        return "MultiDimDeviceArrayViewReadExecution(" +
                "array=" + array +
                ", index=" + index + ")";
    }
}
