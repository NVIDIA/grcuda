/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
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
package com.nvidia.grcuda.array;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.LittleEndianNativeArrayView;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Cached.Shared;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

import java.util.Arrays;

@ExportLibrary(InteropLibrary.class)
public class MultiDimDeviceArray extends AbstractArray implements TruffleObject {

    /** Number of elements in each dimension. */
    private final long[] elementsPerDimension;

    /** Stride in each dimension. */
    private final long[] stridePerDimension;

    /** Total number of elements stored in the array. */
    private final long numElements;

    /** true if data is stored in column-major format (Fortran), false row-major (C). */
    private boolean columnMajor;

    /** Mutable view onto the underlying memory buffer. */
    private final LittleEndianNativeArrayView nativeView;

    public MultiDimDeviceArray(AbstractGrCUDAExecutionContext grCUDAExecutionContext, Type elementType, long[] dimensions,
                               boolean useColumnMajor) {
        super(grCUDAExecutionContext, elementType);
        if (dimensions.length < 2) {
            CompilerDirectives.transferToInterpreter();
            throw new IllegalArgumentException(
                            "MultiDimDeviceArray requires at least two dimension, use DeviceArray instead");
        }
        // check arguments
        long prod = 1;
        for (long n : dimensions) {
            if (n < 1) {
                CompilerDirectives.transferToInterpreter();
                throw new IllegalArgumentException("invalid size of dimension " + n);
            }
            prod *= n;
        }
        this.columnMajor = useColumnMajor;
        this.elementsPerDimension = new long[dimensions.length];
        System.arraycopy(dimensions, 0, this.elementsPerDimension, 0, dimensions.length);
        this.stridePerDimension = computeStride(dimensions, columnMajor);
        this.numElements = prod;
        this.nativeView = grCUDAExecutionContext.getCudaRuntime().cudaMallocManaged(getSizeBytes());
        // Register the array in the GrCUDAExecutionContext;
        this.registerArray();
    }

    private static long[] computeStride(long[] dimensions, boolean columnMajor) {
        long prod = 1;
        long[] stride = new long[dimensions.length];
        if (columnMajor) {
            for (int i = 0; i < dimensions.length; i++) {
                stride[i] = prod;
                prod *= dimensions[i];
            }
        } else {
            for (int i = dimensions.length - 1; i >= 0; i--) {
                stride[i] = prod;
                prod *= dimensions[i];
            }
        }
        return stride;
    }

    public final int getNumberDimensions() {
        return elementsPerDimension.length;
    }

    public final long[] getShape() {
        long[] shape = new long[elementsPerDimension.length];
        System.arraycopy(elementsPerDimension, 0, shape, 0, elementsPerDimension.length);
        return shape;
    }

    public final long getElementsInDimension(int dimension) {
        if (dimension < 0 || dimension >= elementsPerDimension.length) {
            CompilerDirectives.transferToInterpreter();
            throw new IllegalArgumentException("invalid dimension index " + dimension + ", valid [0, " + elementsPerDimension.length + ']');
        }
        return elementsPerDimension[dimension];
    }

    public long getStrideInDimension(int dimension) {
        if (dimension < 0 || dimension >= stridePerDimension.length) {
            CompilerDirectives.transferToInterpreter();
            throw new IllegalArgumentException("invalid dimension index " + dimension + ", valid [0, " + stridePerDimension.length + ']');
        }
        return stridePerDimension[dimension];
    }

    final boolean isIndexValidInDimension(long index, int dimension) {
        long numElementsInDim = getElementsInDimension(dimension);
        return (index > 0) && (index < numElementsInDim);
    }

    final boolean isColumnMajorFormat() {
        return columnMajor;
    }

    long getNumElements() {
        return numElements;
    }

    @Override
    final public long getSizeBytes() {
        return numElements * elementType.getSizeBytes();
    }

    @Override
    public final long getPointer() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return nativeView.getStartAddress();
    }

    final LittleEndianNativeArrayView getNativeView() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return nativeView;
    }

    @Override
    public String toString() {
        return "MultiDimDeviceArray(elementType=" + elementType +
                        ", dims=" + Arrays.toString(elementsPerDimension) +
                        ", Elements=" + numElements +
                        ", size=" + getSizeBytes() + " bytes" +
                        ", nativeView=" + nativeView + ')';
    }

    @Override
    protected void finalize() throws Throwable {
        if (!arrayFreed) {
            grCUDAExecutionContext.getCudaRuntime().cudaFree(nativeView);
        }
        super.finalize();
    }

    @Override
    public void freeMemory() {
        if (arrayFreed) {
            throw new GrCUDAException("device array already freed");
        }
        grCUDAExecutionContext.getCudaRuntime().cudaFree(nativeView);
        arrayFreed = true;
    }

    //
    // Implementation of InteropLibrary
    //

    @ExportMessage
    @SuppressWarnings("static-method")
    @Override
    public long getArraySize() {
        return elementsPerDimension[0];
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    @Override
    boolean isArrayElementReadable(long index) {
        return index >= 0 && index < elementsPerDimension[0];
    }

    @ExportMessage
    @Override
    Object readArrayElement(long index) throws InvalidArrayIndexException {
        if ((index < 0) || (index >= elementsPerDimension[0])) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        long offset = index * stridePerDimension[0];
        return new MultiDimDeviceArrayView(this, 1, offset, stridePerDimension[1]);
    }
}
