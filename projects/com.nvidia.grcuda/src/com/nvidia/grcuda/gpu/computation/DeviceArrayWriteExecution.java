package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.array.DeviceArray;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.profiles.ValueProfile;

public class DeviceArrayWriteExecution extends GrCUDAComputationalElement {

    private final DeviceArray array;
    private final long index;
    private final Object value;
    private final InteropLibrary valueLibrary;
    private final ValueProfile elementTypeProfile;

    public DeviceArrayWriteExecution(DeviceArray array,
                                     long index,
                                     Object value,
                                     InteropLibrary valueLibrary,
                                     ValueProfile elementTypeProfile) {
        super(array.getGrCUDAExecutionContext(), new DeviceArrayExecutionInitializer(array));
        this.array = array;
        this.index = index;
        this.value = value;
        this.valueLibrary = valueLibrary;
        this.elementTypeProfile = elementTypeProfile;
    }

    @Override
    public Object execute() throws UnsupportedTypeException {
        array.writeArrayElementImpl(index, value, valueLibrary, elementTypeProfile);
        this.setComputationFinished();
        return NoneValue.get();
    }

    @Override
    public String toString() {
        return "DeviceArrayWriteExecution(" +
                "array=" + array +
                ", index=" + index +
                ", value=" + value +
                ")";
    }
}
