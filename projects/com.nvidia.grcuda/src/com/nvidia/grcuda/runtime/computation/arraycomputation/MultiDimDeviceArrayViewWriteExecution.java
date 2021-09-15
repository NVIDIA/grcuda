package com.nvidia.grcuda.runtime.computation.arraycomputation;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.runtime.array.MultiDimDeviceArrayView;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.profiles.ValueProfile;

public class MultiDimDeviceArrayViewWriteExecution extends ArrayAccessExecution<MultiDimDeviceArrayView> {

    private final long index;
    private final Object value;
    private final InteropLibrary valueLibrary;
    private final ValueProfile elementTypeProfile;

    public MultiDimDeviceArrayViewWriteExecution(MultiDimDeviceArrayView array,
                                                 long index,
                                                 Object value,
                                                 InteropLibrary valueLibrary,
                                                 ValueProfile elementTypeProfile) {
        super(array.getGrCUDAExecutionContext(), new ArrayAccessExecutionInitializer<>(array.getMdDeviceArray(), false), array);
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
        return "MultiDimDeviceArrayViewReadExecution(" +
                "array=" + array +
                ", index=" + index +
                ", value=" + value +
                ")";
    }
}
