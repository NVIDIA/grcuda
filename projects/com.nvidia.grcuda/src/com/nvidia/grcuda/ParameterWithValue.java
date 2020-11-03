package com.nvidia.grcuda;

import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;

import java.util.Objects;

/**
 * Defines a {@link GrCUDAComputationalElement} argument representing the elements of a NFI signature.
 * For each argument, store its type, if it's a pointer,
 * and if it's constant (i.e. its content cannot be modified in the computation).
 * This class also holds a reference to the actual object associated to the argument;
 */
public class ParameterWithValue extends Parameter {
    private final Object argumentValue;

//    public ParameterWithValue(ArgumentType type, boolean isArray, boolean isConst, Object argumentValue) {
//        super(type, isArray, isConst);
//        this.argumentValue = argumentValue;
//    }

    public ParameterWithValue(String name, Type type, Kind kind, Object argumentValue) {
        super(name, type, kind);
        this.argumentValue = argumentValue;
    }

    public ParameterWithValue(Parameter parameter, Object argumentValue) {
        super(parameter.getPosition(), parameter.getName(), parameter.getType(), parameter.getKind());
        this.argumentValue = argumentValue;
    }

    public Object getArgumentValue() { return this.argumentValue; }

    @Override
    public String toString() {
        return "ComputationArgumentWithValue(" +
                "argumentValue=" + argumentValue +
                ", type=" + type +
                ", isArray=" + isArray +
                ", isConst=" + isConst +
                ')';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ParameterWithValue that = (ParameterWithValue) o;
        return Objects.equals(argumentValue, that.argumentValue);
    }

    @Override
    public int hashCode() {
        return Objects.hash(argumentValue);
    }
}