package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.ArgumentType;
import com.nvidia.grcuda.gpu.computation.ComputationArgumentWithValue;

public class MockArgument extends ComputationArgumentWithValue {
    public MockArgument(Object value) {
        super(ArgumentType.POINTER, true, false, value);
    }

    public MockArgument(Object value, boolean isConst) {
        super(ArgumentType.POINTER, true, isConst, value);
    }

    public MockArgument(Object value, boolean isConst, boolean isArray) {
        super(ArgumentType.POINTER, isArray, isConst, value);
    }
}
