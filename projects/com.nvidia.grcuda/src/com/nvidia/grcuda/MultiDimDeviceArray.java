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
package com.nvidia.grcuda;

import java.util.Arrays;

import com.nvidia.grcuda.DeviceArray.MemberSet;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.LittleEndianNativeArrayView;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Cached.Shared;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

@ExportLibrary(InteropLibrary.class)
public class MultiDimDeviceArray implements TruffleObject {

    private static final MemberSet PUBLIC_MEMBERS = new MemberSet();
    private static final MemberSet MEMBERS = new MemberSet("pointer");

    private final CUDARuntime runtime;

    /** Data type of the elements stored in the array. */
    private final ElementType elementType;

    /** Number of elements in each dimension. */
    private final long[] elementsPerDimension;

    /** Total number of elements stored in the array. */
    private final long totalElementCount;

    /** true if data is stored in column-major format (Fortran), false row-major (C). */
    private boolean columnMajor;

    private final long stride;

    /** Mutable view onto the underlying memory buffer. */
    private final LittleEndianNativeArrayView nativeView;

    public MultiDimDeviceArray(CUDARuntime runtime, ElementType elementType, long[] dimensions,
                    boolean useColumnMajor) {
        if (dimensions.length < 2) {
            throw new IllegalArgumentException(
                            "MultiDimDeviceArray requires at least two dimension, use DeviceArray instead");
        }
        // check arguments
        long prod = 1;
        for (long n : dimensions) {
            if (n < 1) {
                throw new IllegalArgumentException("invalid size of dimension " + n);
            }
            prod *= n;
        }
        this.runtime = runtime;
        this.elementType = elementType;
        this.elementsPerDimension = new long[dimensions.length];
        System.arraycopy(dimensions, 0, this.elementsPerDimension, 0, dimensions.length);
        this.totalElementCount = prod;
        this.columnMajor = useColumnMajor;
        this.stride = computeStrideInDim(0);
        this.nativeView = runtime.cudaMallocManaged(getSizeBytes());
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
            throw new IllegalArgumentException("invalid dimension index " + dimension +
                            ", valid [0, " + elementsPerDimension.length + ']');
        }
        return elementsPerDimension[dimension];
    }

    final boolean isIndexValidInDimension(long index, int dimension) {
        long numElementsInDim = getElementsInDimension(dimension);
        return (index > 0) && (index < numElementsInDim);
    }

    final boolean isColumnMajorFormat() {
        return columnMajor;
    }

    long getTotalElementCount() {
        return totalElementCount;
    }

    final long getSizeBytes() {
        return totalElementCount * elementType.getSizeBytes();
    }

    public final long getPointer() {
        return nativeView.getStartAddress();
    }

    final ElementType getElementType() {
        return elementType;
    }

    final LittleEndianNativeArrayView getNativeView() {
        return nativeView;
    }

    @Override
    public String toString() {
        return "DeviceArray(elementType=" + elementType +
                        ", dims=" + Arrays.toString(elementsPerDimension) +
                        ", Elements=" + totalElementCount +
                        ", size=" + getSizeBytes() + " bytes, " +
                        ", nativeView=" + nativeView + ')';
    }

    @Override
    protected void finalize() throws Throwable {
        runtime.cudaFree(nativeView);
        super.finalize();
    }

    //
    // Implementation of InteropLibrary
    //

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    long getArraySize() {
        return elementsPerDimension[0];
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isArrayElementReadable(long index) {
        return index >= 0 && index < elementsPerDimension[0];
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isArrayElementModifiable(@SuppressWarnings("unused") long index) {
        return false;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isArrayElementInsertable(@SuppressWarnings("unused") long index) {
        return false;
    }

    final long computeStrideInDim(int dim) {
        long prod = 1;
        if (columnMajor) {
            for (int i = 0; i < dim; i++) {
                prod *= elementsPerDimension[i];
            }
        } else {
            for (int i = dim + 1; i < getNumberDimensions(); i++) {
                prod *= elementsPerDimension[i];
            }
        }
        return prod;
    }

    @ExportMessage
    Object readArrayElement(long index) throws InvalidArrayIndexException {
        // System.out.println("MultiDimDeviceArray::readArrayElement(" + index + ')');
        if ((index < 0) || (index >= elementsPerDimension[0])) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        long offset = index * stride;
        long newStride;
        if (columnMajor) {
            newStride = elementsPerDimension[0];
        } else {
            newStride = stride / elementsPerDimension[1];
        }
        return new MultiDimDeviceArrayView(this, 1, offset, newStride);
    }

    @ExportMessage
    void writeArrayElement(@SuppressWarnings("unused") long index, @SuppressWarnings("unused") Object value) {
        throw new IllegalStateException("attempting to write MultiDimensionArray directly");
    }

    @ExportMessage
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    Object getMembers(boolean includeInternal) {
        return includeInternal ? MEMBERS : PUBLIC_MEMBERS;
    }

    @ExportMessage
    boolean isMemberReadable(String member,
                    @Shared("member") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        return "pointer".equals(memberProfile.profile(member));
    }

    @ExportMessage
    Object readMember(String member,
                    @Shared("member") @Cached("createIdentityProfile()") ValueProfile memberProfile) throws UnknownIdentifierException {
        if (!isMemberReadable(member, memberProfile)) {
            CompilerDirectives.transferToInterpreter();
            throw UnknownIdentifierException.create(member);
        }
        return getPointer();
    }

    @ExportMessage
    boolean isPointer() {
        return true;
    }

    @ExportMessage
    long asPointer() {
        return getPointer();
    }
}
