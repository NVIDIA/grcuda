package com.nvidia.grcuda.runtime.computation.arraycomputation;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.oracle.truffle.api.CompilerDirectives;

import java.util.Optional;

/**
 * Computational elements that represents a low-level memory copy from/to a {@link AbstractArray}
 */
public abstract class ArrayCopyFunctionExecution extends GrCUDAComputationalElement {

    /**
     * The {@link AbstractArray} used in the copy;
     */
    protected final AbstractArray array;
    /**
     * Whether this computation copies data from the array or writes to it;
     */
    protected final DeviceArrayCopyFunction.CopyDirection direction;
    /**
     * Number of elements copied (expressed as number of elements, not as a size in bytes);
     */
    protected final long numElements;

    protected boolean isComputationArrayAccess = true;

    //FIXME: create constructor with executioninitiaizer as argument, then in the function that craete this class, use the constructor with iitializer if we have a second device array, else use default initializer

    public ArrayCopyFunctionExecution(AbstractArray array, DeviceArrayCopyFunction.CopyDirection direction, long numElements, ArrayCopyFunctionExecutionInitializer dependencyInitializer) {
        super(array.getGrCUDAExecutionContext(), dependencyInitializer);
        this.array = array;
        this.direction = direction;
        this.numElements = numElements;
    }

    @Override
    public Object execute() {
        if (this.numElements * this.array.getElementType().getSizeBytes() > this.array.getSizeBytes()) {
            CompilerDirectives.transferToInterpreter();
            throw new IndexOutOfBoundsException();
        }
        this.executeInner();
        this.setComputationFinished();
        return NoneValue.get();
    }

    /**
     * Provide different implementations of the copy execution, depending on whether we operate on pointers, arrays, etc.
     */
    abstract void executeInner();

    @Override
    public void updateIsComputationArrayAccess() {
        this.array.setLastComputationArrayAccess(isComputationArrayAccess);
    }

    @Override
    protected Optional<CUDAStream> additionalStreamDependencyImpl() {
        return Optional.of(array.getStreamMapping());
    }

    @Override
    public String toString() {
        return "array copy on " + System.identityHashCode(this.array) + "; direction=" + this.direction + "; size=" + this.numElements;
    }
}

