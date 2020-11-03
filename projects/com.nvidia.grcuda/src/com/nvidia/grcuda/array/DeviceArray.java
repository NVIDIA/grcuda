/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
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
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.LittleEndianNativeArrayView;
import com.nvidia.grcuda.gpu.computation.DeviceArrayReadExecution;
import com.nvidia.grcuda.gpu.computation.DeviceArrayWriteExecution;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Cached.Shared;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

@ExportLibrary(InteropLibrary.class)
public final class DeviceArray extends AbstractArray implements TruffleObject {

    /** Total number of elements stored in the array. */
    private final long numElements;

    /**
     * Total number of bytes allocated and used to store the array data (includes padding).
     */
    private final long sizeBytes;

    /** Mutable view onto the underlying memory buffer. */
    private final LittleEndianNativeArrayView nativeView;

    public DeviceArray(AbstractGrCUDAExecutionContext grCUDAExecutionContext, long numElements, Type elementType) {
        super(grCUDAExecutionContext, elementType);
        this.numElements = numElements;
        this.sizeBytes = numElements * elementType.getSizeBytes();
        this.nativeView = grCUDAExecutionContext.getCudaRuntime().cudaMallocManaged(sizeBytes);
        // Register the array in the GrCUDAExecutionContext;
        this.registerArray();
    }

    @Override
    final public long getSizeBytes() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return sizeBytes;
    }

    @Override
    public long getPointer() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return nativeView.getStartAddress();
    }

    public Type getElementType() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return elementType;
    }

    @Override
    public String toString() {
        if (arrayFreed) {
            return "DeviceArray(memory freed)";
        } else {
            return "DeviceArray(elementType=" + elementType + ", numElements=" + numElements + ", nativeView=" + nativeView + ')';
        }
    }

    @Override
    protected void finalize() throws Throwable {
        if (!arrayFreed) {
            grCUDAExecutionContext.getCudaRuntime().cudaFree(nativeView);
        }
        super.finalize();
    }

//    public void copyFrom(long fromPointer, long numCopyElements) throws IndexOutOfBoundsException {
//        if (arrayFreed) {
//            CompilerDirectives.transferToInterpreter();
//            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
//        }
//        long numBytesToCopy = numCopyElements * elementType.getSizeBytes();
//        if (numBytesToCopy > getSizeBytes()) {
//            CompilerDirectives.transferToInterpreter();
//            throw new IndexOutOfBoundsException();
//        }
//        runtime.cudaMemcpy(getPointer(), fromPointer, numBytesToCopy);
//    }
//
//    public void copyTo(long toPointer, long numCopyElements) throws IndexOutOfBoundsException {
//        if (arrayFreed) {
//            CompilerDirectives.transferToInterpreter();
//            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
//        }
//        long numBytesToCopy = numCopyElements * elementType.getSizeBytes();
//        if (numBytesToCopy > getSizeBytes()) {
//            CompilerDirectives.transferToInterpreter();
//            throw new IndexOutOfBoundsException();
//        }
//        runtime.cudaMemcpy(toPointer, getPointer(), numBytesToCopy);
//    }

    @Override
    public void freeMemory() {
        if (arrayFreed) {
            throw new GrCUDAException("device array already freed");
        }
        grCUDAExecutionContext.getCudaRuntime().cudaFree(nativeView);
        arrayFreed = true;
    }

    // Implementation of InteropLibrary

    @ExportMessage
    public long getArraySize() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return numElements;
    }

    @ExportMessage
    boolean isArrayElementReadable(long index) {
        return !arrayFreed && index >= 0 && index < numElements;
    }

    @ExportMessage
    boolean isArrayElementModifiable(long index) {
        return index >= 0 && index < numElements;
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    boolean isArrayElementInsertable(@SuppressWarnings("unused") long index) {
        return false;
    }

    @ExportMessage
    Object readArrayElement(long index,
                    @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws InvalidArrayIndexException {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        if ((index < 0) || (index >= numElements)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        try {
            if (this.canSkipScheduling()) {
                // Fast path, skip the DAG scheduling;
                return readArrayElementImpl(index, elementTypeProfile);
            } else {
                return new DeviceArrayReadExecution(this, index, elementTypeProfile).schedule();
            }
        } catch (UnsupportedTypeException e) {
            e.printStackTrace();
            return null;
        }
    }

    public Object readArrayElementImpl(long index, ValueProfile elementTypeProfile) {
        switch (elementTypeProfile.profile(elementType)) {
            case CHAR:
                return nativeView.getByte(index);
            case SINT16:
                return nativeView.getShort(index);
            case SINT32:
                return nativeView.getInt(index);
            case SINT64:
                return nativeView.getLong(index);
            case FLOAT:
                return nativeView.getFloat(index);
            case DOUBLE:
                return nativeView.getDouble(index);
        }
        return null;
    }

    @ExportMessage
    public void writeArrayElement(long index, Object value,
                    @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
                    @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws UnsupportedTypeException, InvalidArrayIndexException {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        if ((index < 0) || (index >= numElements)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        if (this.canSkipScheduling()) {
            // Fast path, skip the DAG scheduling;
            writeArrayElementImpl(index, value, valueLibrary, elementTypeProfile);
        } else {
            new DeviceArrayWriteExecution(this, index, value, valueLibrary, elementTypeProfile).schedule();
        }
    }

    public void writeArrayElementImpl(long index, Object value,
                                      InteropLibrary valueLibrary,
                                      ValueProfile elementTypeProfile) throws UnsupportedTypeException {
        try {
            switch (elementTypeProfile.profile(elementType)) {
                case CHAR:
                    nativeView.setByte(index, valueLibrary.asByte(value));
                    break;
                case SINT16:
                    nativeView.setShort(index, valueLibrary.asShort(value));
                    break;
                case SINT32:
                    nativeView.setInt(index, valueLibrary.asInt(value));
                    break;
                case SINT64:
                    nativeView.setLong(index, valueLibrary.asLong(value));
                    break;
                case FLOAT:
                    // going via "double" to allow floats to be initialized with doubles
                    nativeView.setFloat(index, (float) valueLibrary.asDouble(value));
                    break;
                case DOUBLE:
                    nativeView.setDouble(index, valueLibrary.asDouble(value));
                    break;
            }
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{value}, "value cannot be coerced to " + elementType);
        }
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isPointer() {
        return true;
    }

    @ExportMessage
    long asPointer() {
        return getPointer();
    }
}
