package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.array.MultiDimDeviceArrayView;
import com.oracle.truffle.api.profiles.ValueProfile;

public class MultiDimDeviceArrayViewReadExecution extends GrCUDAComputationalElement {

    private final MultiDimDeviceArrayView array;
    private final long index;
    private final ValueProfile elementTypeProfile;

    public MultiDimDeviceArrayViewReadExecution(MultiDimDeviceArrayView array,
                                                long index,
                                                ValueProfile elementTypeProfile) {
        super(array.getGrCUDAExecutionContext(), new ArrayExecutionInitializer(array.getMdDeviceArray()));
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
        return "MultiDimDeviceArrayViewReadExecution(" +
                "array=" + array +
                ", index=" + index + ")";
    }
}
