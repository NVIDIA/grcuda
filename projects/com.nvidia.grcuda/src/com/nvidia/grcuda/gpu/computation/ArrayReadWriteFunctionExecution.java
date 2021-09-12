package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.Optional;

/**
 * Computational elements that represents a low-level memory copy from/to a {@link AbstractArray}
 */
public abstract class ArrayReadWriteFunctionExecution extends GrCUDAComputationalElement {

    /**
     * The {@link AbstractArray} used in the copy;
     */
    protected final AbstractArray array;
    /**
     * Whether this computations copies data from the array or writes to it;
     */
    protected final DeviceArrayCopyFunction.CopyDirection direction;
    /**
     * Number of elements copied (expressed as number of elements, not as a size in bytes);
     */
    protected final long numElements;

    protected boolean isComputationArrayAccess = true;

    public ArrayReadWriteFunctionExecution(AbstractArray array, DeviceArrayCopyFunction.CopyDirection direction, long numElements) {
        super(array.getGrCUDAExecutionContext(), new ArrayExecutionInitializer<>(array, direction.equals(DeviceArrayCopyFunction.CopyDirection.TO_POINTER)));
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
    protected Optional<CUDAStream> additionalStreamDependencyImpl() { return Optional.of(this.array.getStreamMapping()); }

    @Override
    public String toString() {
        return "array copy on " + System.identityHashCode(this.array) + "; direction=" + this.direction + "; size=" + this.numElements;
    }
}

