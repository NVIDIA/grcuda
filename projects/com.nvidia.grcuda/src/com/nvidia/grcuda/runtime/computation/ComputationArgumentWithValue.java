package com.nvidia.grcuda.runtime.computation;

import com.nvidia.grcuda.Type;

import java.util.Objects;

/**
 * Defines a {@link GrCUDAComputationalElement} argument representing the elements of a NFI signature.
 * For each argument, store its type, if it's a pointer,
 * and if it's constant (i.e. its content cannot be modified in the computation).
 * This class also holds a reference to the actual object associated to the argument;
 */
public class ComputationArgumentWithValue extends ComputationArgument {
    private final Object argumentValue;

    public ComputationArgumentWithValue(String name, Type type, Kind kind, Object argumentValue) {
        super(name, type, kind);
        this.argumentValue = argumentValue;
    }

    public ComputationArgumentWithValue(ComputationArgument computationArgument, Object argumentValue) {
        super(computationArgument.getPosition(), computationArgument.getName(), computationArgument.getType(), computationArgument.getKind());
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
        ComputationArgumentWithValue that = (ComputationArgumentWithValue) o;
        return Objects.equals(argumentValue, that.argumentValue);
    }

    @Override
    public int hashCode() {
        return Objects.hash(argumentValue);
    }
}