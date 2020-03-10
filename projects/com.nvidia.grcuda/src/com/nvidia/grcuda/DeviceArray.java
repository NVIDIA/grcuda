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
package com.nvidia.grcuda;

import java.util.Arrays;

import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.LittleEndianNativeArrayView;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.CompilationFinal;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
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

@ExportLibrary(InteropLibrary.class)
public final class DeviceArray implements TruffleObject {

    private static final String POINTER = "pointer";
    private static final String COPY_FROM = "copyFrom";
    private static final String COPY_TO = "copyTo";

    private static final MemberSet PUBLIC_MEMBERS = new MemberSet(COPY_FROM, COPY_TO);
    private static final MemberSet MEMBERS = new MemberSet(POINTER, COPY_FROM, COPY_TO);

    @ExportLibrary(InteropLibrary.class)
    public static final class MemberSet implements TruffleObject {

        @CompilationFinal(dimensions = 1) private final String[] values;

        public MemberSet(String... values) {
            this.values = values;
        }

        @ExportMessage
        @SuppressWarnings("static-method")
        public boolean hasArrayElements() {
            return true;
        }

        @ExportMessage
        public long getArraySize() {
            return values.length;
        }

        @ExportMessage
        public boolean isArrayElementReadable(long index) {
            return index >= 0 && index < values.length;
        }

        @ExportMessage
        public Object readArrayElement(long index) throws InvalidArrayIndexException {
            if ((index < 0) || (index >= values.length)) {
                CompilerDirectives.transferToInterpreter();
                throw InvalidArrayIndexException.create(index);
            }
            return values[(int) index];
        }

        @TruffleBoundary
        public boolean constainsValue(String name) {
            return Arrays.asList(values).contains(name);
        }
    }

    private final CUDARuntime runtime;

    /** Data type of elements stored in the array. */
    private final ElementType elementType;

    /** Total number of elements stored in the array. */
    private final long numElements;

    /**
     * Total number of bytes allocated and used to store the array data (includes padding).
     */
    private final long sizeBytes;

    /** Mutable view onto the underlying memory buffer. */
    private final LittleEndianNativeArrayView nativeView;

    public DeviceArray(CUDARuntime runtime, long numElements, ElementType elementType) {
        this.runtime = runtime;
        this.numElements = numElements;
        this.elementType = elementType;
        this.sizeBytes = numElements * elementType.getSizeBytes();
        this.nativeView = runtime.cudaMallocManaged(this.sizeBytes);
    }

    long getSizeBytes() {
        return sizeBytes;
    }

    public long getPointer() {
        return nativeView.getStartAddress();
    }

    public ElementType getElementType() {
        return elementType;
    }

    @Override
    public String toString() {
        return "DeviceArray(elementType=" + elementType + ", numElements=" + numElements + ", nativeView=" + nativeView + ')';
    }

    @Override
    protected void finalize() throws Throwable {
        runtime.cudaFree(nativeView);
        super.finalize();
    }

    // Implementation of InteropLibrary

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    public long getArraySize() {
        return numElements;
    }

    @ExportMessage
    boolean isArrayElementReadable(long index) {
        return index >= 0 && index < numElements;
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

        if ((index < 0) || (index >= numElements)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);

        }
        switch (elementTypeProfile.profile(elementType)) {
            case BYTE:
            case CHAR:
                return nativeView.getByte(index);
            case SHORT:
                return nativeView.getShort(index);
            case INT:
                return nativeView.getInt(index);
            case LONG:
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

        if ((index < 0) || (index >= numElements)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        try {
            switch (elementTypeProfile.profile(elementType)) {

                case BYTE:
                case CHAR:
                    nativeView.setByte(index, valueLibrary.asByte(value));
                    break;
                case SHORT:
                    nativeView.setShort(index, valueLibrary.asShort(value));
                    break;
                case INT:
                    nativeView.setInt(index, valueLibrary.asInt(value));
                    break;
                case LONG:
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
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    Object getMembers(boolean includeInternal) {
        return includeInternal ? MEMBERS : PUBLIC_MEMBERS;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String memberName,
                    @Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        String name = memberProfile.profile(memberName);
        return POINTER.equals(name) || COPY_FROM.equals(name) || COPY_TO.equals(name);
    }

    @ExportMessage
    Object readMember(String memberName,
                    @Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) throws UnknownIdentifierException {
        if (!isMemberReadable(memberName, memberProfile)) {
            CompilerDirectives.transferToInterpreter();
            throw UnknownIdentifierException.create(memberName);
        }
        if (POINTER.equals(memberName)) {
            return getPointer();
        }
        if (COPY_FROM.equals(memberName)) {
            return new DeviceArrayCopyFunction(this, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
        }
        if (COPY_TO.equals(memberName)) {
            return new DeviceArrayCopyFunction(this, DeviceArrayCopyFunction.CopyDirection.TO_POINTER);
        }
        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(memberName);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberInvocable(String memberName) {
        return COPY_FROM.equals(memberName) || COPY_TO.equals(memberName);
    }

    @ExportMessage
    Object invokeMember(String memberName,
                    Object[] arguments,
                    @CachedLibrary("this") InteropLibrary interopRead,
                    @CachedLibrary(limit = "1") InteropLibrary interopExecute)
                    throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        return interopExecute.execute(interopRead.readMember(this, memberName), arguments);
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

    public void copyFrom(long fromPointer, long numCopyElements) throws IndexOutOfBoundsException {
        long numBytesToCopy = numCopyElements * elementType.getSizeBytes();
        if (numBytesToCopy > getSizeBytes()) {
            CompilerDirectives.transferToInterpreter();
            throw new IndexOutOfBoundsException();
        }
        runtime.cudaMemcpy(getPointer(), fromPointer, numBytesToCopy);
    }

    public void copyTo(long toPointer, long numCopyElements) throws IndexOutOfBoundsException {
        long numBytesToCopy = numCopyElements * elementType.getSizeBytes();
        if (numBytesToCopy > getSizeBytes()) {
            CompilerDirectives.transferToInterpreter();
            throw new IndexOutOfBoundsException();
        }
        runtime.cudaMemcpy(toPointer, getPointer(), numBytesToCopy);
    }
}
