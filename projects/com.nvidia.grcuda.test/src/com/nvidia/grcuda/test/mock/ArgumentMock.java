package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.ArgumentType;
import com.nvidia.grcuda.gpu.computation.ComputationArgumentWithValue;

public class ArgumentMock extends ComputationArgumentWithValue {
    public ArgumentMock(Object value) {
        super(ArgumentType.POINTER, true, false, value);
    }

    public ArgumentMock(Object value, boolean isConst) {
        super(ArgumentType.POINTER, true, isConst, value);
    }

    public ArgumentMock(Object value, boolean isConst, boolean isArray) {
        super(ArgumentType.POINTER, isArray, isConst, value);
    }
}
