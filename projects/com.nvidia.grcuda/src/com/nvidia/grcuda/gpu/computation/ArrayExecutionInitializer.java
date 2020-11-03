package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.Parameter;
import com.nvidia.grcuda.ParameterWithValue;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.array.AbstractArray;

import java.util.Collections;
import java.util.List;

/**
 * The only argument in {@link com.nvidia.grcuda.array.AbstractArray} computations is the array itself.
 * Note that in {@link com.nvidia.grcuda.array.MultiDimDeviceArrayView} the array is the parent {@link com.nvidia.grcuda.array.MultiDimDeviceArray},
 * while in {@link com.nvidia.grcuda.array.MultiDimDeviceArray} there is currently no need to explicitly represent computations,
 * as they cannot directly the underlying memory;
 */
class ArrayExecutionInitializer<T extends AbstractArray> implements InitializeArgumentList {

    private final T array;
    private final boolean readOnly;
    private final static String PARAMETER_NAME = "array_access";

    ArrayExecutionInitializer(T array) {
        this(array, false);
    }

    ArrayExecutionInitializer(T array, boolean readOnly) {
        this.array = array;
        this.readOnly = readOnly;
    }

    @Override
    public List<ParameterWithValue> initialize() {
        return Collections.singletonList(
                new ParameterWithValue(PARAMETER_NAME, Type.NFI_POINTER, this.readOnly ? Parameter.Kind.POINTER_IN : Parameter.Kind.POINTER_INOUT, this.array));
    }
}
