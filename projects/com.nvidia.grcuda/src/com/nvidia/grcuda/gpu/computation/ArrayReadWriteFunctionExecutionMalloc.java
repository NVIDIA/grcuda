package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.oracle.truffle.api.CompilerDirectives;

/**
 * Fastest {@link AbstractArray} memcpy implementation, it operates using a cudaMemcpy directly on a native pointer.
 * This implementation is used when copying data between AbstractArrays, or when copying data from/to an array backed
 * by native memory, such as numpy arrays;
 */
public class ArrayReadWriteFunctionExecutionMalloc extends ArrayReadWriteFunctionExecution {
    /**
     * A memory pointer from which data copied to the array are retrieved, or memory pointer to which data are written;
     */
    private final long pointer;

    public ArrayReadWriteFunctionExecutionMalloc(AbstractArray array, DeviceArrayCopyFunction.CopyDirection direction, long numElements, long pointer) {
        super(array, direction, numElements);
        this.pointer = pointer;
    }

    @Override
    void executeInner() {
        long numBytesToCopy = this.numElements * this.array.getElementType().getSizeBytes();
        long fromPointer;
        long destPointer;
        if (direction == DeviceArrayCopyFunction.CopyDirection.FROM_POINTER) {
            fromPointer = pointer;
            destPointer = array.getPointer();
        } else if (direction == DeviceArrayCopyFunction.CopyDirection.TO_POINTER) {
            fromPointer = array.getPointer();
            destPointer = pointer;
        } else {
            CompilerDirectives.transferToInterpreter();
            throw new DeviceArrayCopyException("invalid direction for copy: " + direction);
        }
        // If the array visibility is restricted to a stream, provide the stream to memcpy;
        if (array.getStreamMapping().isDefaultStream()) {
            grCUDAExecutionContext.getCudaRuntime().cudaMemcpy(destPointer, fromPointer, numBytesToCopy);
        } else {
            grCUDAExecutionContext.getCudaRuntime().cudaMemcpy(destPointer, fromPointer, numBytesToCopy, array.getStreamMapping());
        }
    }

    @Override
    public String toString() {
        return "array memcpy on " + System.identityHashCode(array) + "; direction=" + direction + "; target=" + pointer + "; size=" + numElements;
    }
}
