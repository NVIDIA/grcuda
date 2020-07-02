package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.Optional;

/**
 * Computational elements that represents a low-level memory copy from/to a {@link AbstractArray}
 */
public class ArrayReadWriteFunctionExecution extends GrCUDAComputationalElement {

    /**
     * The {@link AbstractArray} used in the copy;
     */
    private final AbstractArray array;
    /**
     * Whether this computations copies data from the array or writes to it;
     */
    private final DeviceArrayCopyFunction.CopyDirection direction;
    /**
     * A memory pointer from which data copied to the array are retrieved, or memory pointer to which data are written;
     */
    private final long pointer;
    /**
     * Number of elements copied (expressed as number of elements, not as a size in bytes);
     */
    private final long numElements;

    protected boolean isComputationArrayAccess = true;

    public ArrayReadWriteFunctionExecution(AbstractArray array, DeviceArrayCopyFunction.CopyDirection direction, long pointer, long numElements) {
        super(array.getGrCUDAExecutionContext(), new ArrayExecutionInitializer<>(array, direction.equals(DeviceArrayCopyFunction.CopyDirection.TO_POINTER)));
        this.array = array;
        this.direction = direction;
        this.pointer = pointer;
        this.numElements = numElements;
    }

    @Override
    public Object execute() throws UnsupportedTypeException {
        if (direction == DeviceArrayCopyFunction.CopyDirection.FROM_POINTER) {
            copyFrom();
        }
        if (direction == DeviceArrayCopyFunction.CopyDirection.TO_POINTER) {
            copyTo();
        }
        this.setComputationFinished();
        return NoneValue.get();
    }

    private void copyFrom() throws IndexOutOfBoundsException {
        long numBytesToCopy = numElements * array.getElementType().getSizeBytes();
        if (numBytesToCopy > array.getSizeBytes()) {
            CompilerDirectives.transferToInterpreter();
            throw new IndexOutOfBoundsException();
        }
        // If the array visibility is restricted to a stream, provide the stream to memcpy;
        if (array.getStreamMapping().isDefaultStream()) {
            grCUDAExecutionContext.getCudaRuntime().cudaMemcpy(array.getPointer(), pointer, numBytesToCopy);
        } else {
            grCUDAExecutionContext.getCudaRuntime().cudaMemcpy(array.getPointer(), pointer, numBytesToCopy, array.getStreamMapping());
        }
    }

    private void copyTo() throws IndexOutOfBoundsException {
        long numBytesToCopy = numElements * array.getElementType().getSizeBytes();
        if (numBytesToCopy > array.getSizeBytes()) {
            CompilerDirectives.transferToInterpreter();
            throw new IndexOutOfBoundsException();
        }
        // If the array visibility is restricted to a stream, provide the stream to memcpy; 
        if (array.getStreamMapping().isDefaultStream()) {
            grCUDAExecutionContext.getCudaRuntime().cudaMemcpy(pointer, array.getPointer(), numBytesToCopy);
        } else {
            grCUDAExecutionContext.getCudaRuntime().cudaMemcpy(pointer, array.getPointer(), numBytesToCopy, array.getStreamMapping());
        }
    }

    @Override
    public void updateIsComputationArrayAccess() {
        this.array.setLastComputationArrayAccess(isComputationArrayAccess);
    }

    @Override
    protected Optional<CUDAStream> additionalStreamDependencyImpl() { return Optional.of(array.getStreamMapping()); }

    @Override
    public String toString() {
        return "array memcpy on " + System.identityHashCode(array) + "; direction=" + direction + "; target=" + pointer + "; size=" + numElements;
    }
}
