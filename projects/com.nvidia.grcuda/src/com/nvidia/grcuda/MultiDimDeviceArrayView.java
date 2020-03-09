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
public final class MultiDimDeviceArrayView implements TruffleObject {

    private final MultiDimDeviceArray mdDeviceArray;
    private final int thisDimension;
    private final long offset;
    private final long stride;

    MultiDimDeviceArrayView(MultiDimDeviceArray mdDeviceArray, int dim, long offset, long stride) {
        this.mdDeviceArray = mdDeviceArray;
        this.thisDimension = dim;
        this.offset = offset;
        this.stride = stride;
    }

    public int getDimension() {
        return thisDimension;
    }

    public long getOffset() {
        return offset;
    }

    public long getStride() {
        return stride;
    }

    @Override
    public String toString() {
        return String.format("MultiDimDeviceArrayView(dim=%d, offset=%d, stride=%d)\n",
                        thisDimension, offset, stride);
    }

    //
    // Implementation of Interop Library
    //

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    long getArraySize() {
        return mdDeviceArray.getElementsInDimension(thisDimension);
    }

    @ExportMessage
    boolean isArrayElementReadable(long index) {
        return index >= 0 && index < mdDeviceArray.getElementsInDimension(thisDimension);
    }

    @ExportMessage
    boolean isArrayElementModifiable(long index) {
        return (thisDimension + 1) == mdDeviceArray.getNumberDimensions() &&
                        index >= 0 && index < mdDeviceArray.getElementsInDimension(thisDimension);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isArrayElementInsertable(@SuppressWarnings("unused") long index) {
        return false;
    }

    @ExportMessage
    Object readArrayElement(long index,
                    @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws InvalidArrayIndexException {
        if ((index < 0) || (index >= mdDeviceArray.getElementsInDimension(thisDimension))) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        if ((thisDimension + 1) == mdDeviceArray.getNumberDimensions()) {
            long flatIndex = offset + index * stride;
            switch (elementTypeProfile.profile(mdDeviceArray.getElementType())) {
                case BYTE:
                case CHAR:
                    return mdDeviceArray.getNativeView().getByte(flatIndex);
                case SHORT:
                    return mdDeviceArray.getNativeView().getShort(flatIndex);
                case INT:
                    return mdDeviceArray.getNativeView().getInt(flatIndex);
                case LONG:
                    return mdDeviceArray.getNativeView().getLong(flatIndex);
                case FLOAT:
                    return mdDeviceArray.getNativeView().getFloat(flatIndex);
                case DOUBLE:
                    return mdDeviceArray.getNativeView().getDouble(flatIndex);
            }
            return null;
        } else {
            long off = offset + index * stride;
            long newStride = mdDeviceArray.getStrideInDimension(thisDimension + 1);
            return new MultiDimDeviceArrayView(mdDeviceArray, thisDimension + 1, off, newStride);
        }
    }

    @ExportMessage
    void writeArrayElement(long index, Object value,
                    @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
                    @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws UnsupportedTypeException, InvalidArrayIndexException {
        if ((index < 0) || (index >= mdDeviceArray.getElementsInDimension(thisDimension))) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        if ((thisDimension + 1) == mdDeviceArray.getNumberDimensions()) {
            long flatIndex = offset + index * stride;
            try {
                switch (elementTypeProfile.profile(mdDeviceArray.getElementType())) {
                    case BYTE:
                    case CHAR:
                        mdDeviceArray.getNativeView().setByte(flatIndex, valueLibrary.asByte(value));
                        break;
                    case SHORT:
                        mdDeviceArray.getNativeView().setShort(flatIndex, valueLibrary.asShort(value));
                        break;
                    case INT:
                        mdDeviceArray.getNativeView().setInt(flatIndex, valueLibrary.asInt(value));
                        break;
                    case LONG:
                        mdDeviceArray.getNativeView().setLong(flatIndex, valueLibrary.asLong(value));
                        break;
                    case FLOAT:
                        // InteropLibrary does not downcast Double to Float due loss of precision
                        mdDeviceArray.getNativeView().setFloat(flatIndex, (float) valueLibrary.asDouble(value));
                        break;
                    case DOUBLE:
                        mdDeviceArray.getNativeView().setDouble(flatIndex, valueLibrary.asDouble(value));
                        break;
                }
            } catch (UnsupportedMessageException e) {
                CompilerDirectives.transferToInterpreter();
                throw UnsupportedTypeException.create(new Object[]{value}, "value cannot be coerced to " + mdDeviceArray.getElementType());
            }
        } else {
            CompilerDirectives.transferToInterpreter();
            throw new IllegalStateException("tried to write non-last dimension in MultiDimDeviceArrayView");
        }
    }
}
