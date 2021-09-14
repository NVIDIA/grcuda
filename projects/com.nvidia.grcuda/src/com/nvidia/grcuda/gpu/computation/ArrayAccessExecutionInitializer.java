package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.ComputationArgument;
import com.nvidia.grcuda.ComputationArgumentWithValue;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.array.AbstractArray;

import java.util.Collections;
import java.util.List;

/**
 * The only argument in {@link com.nvidia.grcuda.array.AbstractArray} computations is the array itself.
 * Note that in {@link com.nvidia.grcuda.array.MultiDimDeviceArrayView} the array is the parent {@link com.nvidia.grcuda.array.MultiDimDeviceArray},
 * while in {@link com.nvidia.grcuda.array.MultiDimDeviceArray} there is currently no need to explicitly represent computations,
 * as they cannot directly access the underlying memory;
 */
class ArrayAccessExecutionInitializer<T extends AbstractArray> implements InitializeDependencyList {

    private final T array;
    private final boolean readOnly;
    private final static String PARAMETER_NAME = "array_access";

    ArrayAccessExecutionInitializer(T array, boolean readOnly) {
        this.array = array;
        this.readOnly = readOnly;
    }

    @Override
    public List<ComputationArgumentWithValue> initialize() {
        return Collections.singletonList(
                new ComputationArgumentWithValue(PARAMETER_NAME, Type.NFI_POINTER, this.readOnly ? ComputationArgument.Kind.POINTER_IN : ComputationArgument.Kind.POINTER_INOUT, this.array));
    }
}
