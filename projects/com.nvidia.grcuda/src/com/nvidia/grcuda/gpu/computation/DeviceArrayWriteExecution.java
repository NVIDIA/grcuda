package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.array.DeviceArray;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.profiles.ValueProfile;

public class DeviceArrayWriteExecution extends ArrayAccessExecution<DeviceArray> {

    private final long index;
    private final Object value;
    private final InteropLibrary valueLibrary;
    private final ValueProfile elementTypeProfile;

    public DeviceArrayWriteExecution(DeviceArray array,
                                     long index,
                                     Object value,
                                     InteropLibrary valueLibrary,
                                     ValueProfile elementTypeProfile) {
        super(array.getGrCUDAExecutionContext(), new ArrayExecutionInitializer<>(array, false), array);
        this.index = index;
        this.value = value;
        this.valueLibrary = valueLibrary;
        this.elementTypeProfile = elementTypeProfile;
    }

    @Override
    public Object execute() throws UnsupportedTypeException {
        array.writeNativeView(index, value, valueLibrary, elementTypeProfile);
        this.setComputationFinished();
        return NoneValue.get();
    }

    @Override
    public String toString() {
//        return "DeviceArrayWriteExecution(" +
//                "array=" + array +
//                ", index=" + index +
//                ", value=" + value +
//                ")";
        return "array write on " + System.identityHashCode(array) + "; index=" + index + "; value=" + value + "; stream=" + getStream().getStreamNumber();
    }
}
