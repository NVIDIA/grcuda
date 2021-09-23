/*
 * Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.nvidia.grcuda.runtime.computation.arraycomputation;

import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.oracle.truffle.api.CompilerDirectives;

/**
 * Fastest {@link AbstractArray} memcpy implementation, it operates using a cudaMemcpy directly on a native pointer.
 * This implementation is used when copying data between AbstractArrays, or when copying data from/to an array backed
 * by native memory, such as numpy arrays;
 */
public class ArrayCopyFunctionExecutionMemcpy extends ArrayCopyFunctionExecution {
    /**
     * A memory pointer from which data copied to the array are retrieved, or memory pointer to which data are written;
     */
    private final long pointer;

    public ArrayCopyFunctionExecutionMemcpy(AbstractArray array, DeviceArrayCopyFunction.CopyDirection direction, long numElements, long pointer, ArrayCopyFunctionExecutionInitializer dependencyInitializer) {
        super(array, direction, numElements, dependencyInitializer);
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
