package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;

public class ArgumentMock extends ComputationArgumentWithValue {
    public ArgumentMock(Object value) {
        super("argument_mock_nonconst", Type.NFI_POINTER, Kind.POINTER_INOUT, value);
    }

    public ArgumentMock(Object value, boolean isConst) {
        super(isConst ? "argument_mock_const" : "argument_mock_nonconst", Type.NFI_POINTER, isConst ? Kind.POINTER_IN : Kind.POINTER_INOUT, value);
    }

    @Override
    public String toString() {
        return this.getArgumentValue().toString() + (isArray ? "" : " - scalar") + (isConst ? " - const" : "");
    }
}
