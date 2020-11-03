package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.ParameterWithValue;

public class ArgumentMock extends ParameterWithValue {
    public ArgumentMock(Object value) {
        super("argument_mock_nonconst", Type.NFI_POINTER, Kind.POINTER_INOUT, value);
    }

    public ArgumentMock(Object value, boolean isConst) {
        super(isConst ? "argument_mock_const" : "argument_mock_nonconst", Type.NFI_POINTER, isConst ? Kind.POINTER_IN : Kind.POINTER_INOUT, value);
    }

//    public ArgumentMock(Object value, boolean isConst, boolean isArray) {
//        super(ArgumentType.POINTER, isArray, isConst, value);
//    }

    @Override
    public String toString() {
        return this.getArgumentValue().toString() + (isArray ? "" : " - scalar") + (isConst ? " - const" : "");
    }
}
