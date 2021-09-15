package com.nvidia.grcuda.runtime.computation.arraycomputation;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.computation.InitializeDependencyList;

import java.util.ArrayList;
import java.util.List;

public class ArrayCopyFunctionExecutionInitializer implements InitializeDependencyList {

    private final AbstractArray array;
    private final Object otherArray;
    private final DeviceArrayCopyFunction.CopyDirection direction;
    private final static String PARAMETER_NAME_1 = "array_copy_function_arg_1";
    private final static String PARAMETER_NAME_2 = "array_copy_function_arg_2";

    public ArrayCopyFunctionExecutionInitializer(AbstractArray array, Object otherArray, DeviceArrayCopyFunction.CopyDirection direction) {
        this.array = array;
        this.direction = direction;
        this.otherArray = otherArray;
    }

    @Override
    public List<ComputationArgumentWithValue> initialize() {
        ArrayList<ComputationArgumentWithValue> dependencyList = new ArrayList<>();
        dependencyList.add(new ComputationArgumentWithValue(PARAMETER_NAME_1, Type.NFI_POINTER,
                        this.direction.equals(DeviceArrayCopyFunction.CopyDirection.FROM_POINTER) ? ComputationArgument.Kind.POINTER_OUT : ComputationArgument.Kind.POINTER_IN, this.array));
        // If we are copying from/to another DeviceArray, that's also a dependency;
        if (otherArray instanceof AbstractArray) {
            dependencyList.add(new ComputationArgumentWithValue(PARAMETER_NAME_2, Type.NFI_POINTER,
                            this.direction.equals(DeviceArrayCopyFunction.CopyDirection.FROM_POINTER) ? ComputationArgument.Kind.POINTER_IN : ComputationArgument.Kind.POINTER_OUT, this.otherArray));
        }
        return dependencyList;
    }
}
