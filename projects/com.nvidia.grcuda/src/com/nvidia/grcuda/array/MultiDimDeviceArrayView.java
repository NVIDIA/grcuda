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
import com.nvidia.grcuda.gpu.computation.MultiDimDeviceArrayViewReadExecution;
import com.nvidia.grcuda.gpu.computation.MultiDimDeviceArrayViewWriteExecution;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
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
public final class MultiDimDeviceArrayView extends AbstractArray implements TruffleObject {

    private final MultiDimDeviceArray mdDeviceArray;
    private final int thisDimension;
    private final long offset;
    private final long stride;

    MultiDimDeviceArrayView(MultiDimDeviceArray mdDeviceArray, int dim, long offset, long stride) {
        super(mdDeviceArray.grCUDAExecutionContext, mdDeviceArray.elementType, mdDeviceArray.isLastComputationArrayAccess());
        this.mdDeviceArray = mdDeviceArray;
        this.thisDimension = dim;
        this.offset = offset; // Index at which this array view starts;
        this.stride = stride;
        // Register the array in the GrCUDAExecutionContext;
        this.registerArray();
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

    // FIXME: when column-major, low-level memcpy is accessing columns instead of rows!
    @Override
    public final long getPointer() {
        return mdDeviceArray.getPointer() + offset * elementType.getSizeBytes();
    }

    /**
     * Propagate the flag to the parent array, so other temporary views are aware of this computation;
     * @param lastComputationArrayAccess if the last computation on this array is a host read/write
     */
    @Override
    public void setLastComputationArrayAccess(boolean lastComputationArrayAccess) {
        super.setLastComputationArrayAccess(lastComputationArrayAccess);
        this.mdDeviceArray.setLastComputationArrayAccess(lastComputationArrayAccess);
    }

    @Override
    public boolean isLastComputationArrayAccess() { return this.mdDeviceArray.isLastComputationArrayAccess(); }

    /**
     * Propagate the stream mapping to the parent array, so other temporary views are aware of this mapping;
     * @param streamMapping the stream to which this array is associated
     */
    @Override
    public void setStreamMapping(CUDAStream streamMapping) {
        this.mdDeviceArray.setStreamMapping(streamMapping);
        this.streamMapping = streamMapping;
    }

    /**
     * Return the parent stream mapping, to guarantee that all views have the same mapping;
     * @return the stream to which this array is associated
     */
    @Override
    public CUDAStream getStreamMapping() {
        return this.mdDeviceArray.getStreamMapping();
    }

    @Override
    final public long getSizeBytes() {
        return mdDeviceArray.getElementsInDimension(thisDimension) * elementType.getSizeBytes();
    }

    @Override
    public void freeMemory() {
        // This should not be called directly on a view;
        CompilerDirectives.transferToInterpreter();
        throw new GrCUDAException("Freeing memory directly on a MultiDimDeviceArrayView is not allowed");
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
    @Override
    public long getArraySize() {
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
        try {
            if (this.canSkipScheduling()) {
                // Fast path, skip the DAG scheduling;
                return readArrayElementImpl(index, elementTypeProfile);
            } else {
                return new MultiDimDeviceArrayViewReadExecution(this, index, elementTypeProfile).schedule();
            }
        } catch (UnsupportedTypeException e) {
            e.printStackTrace();
            return null;
        }
    }

    public Object readArrayElementImpl(long index, ValueProfile elementTypeProfile) {
        if ((thisDimension + 1) == mdDeviceArray.getNumberDimensions()) {
            long flatIndex = offset + index * stride;
            switch (elementTypeProfile.profile(mdDeviceArray.getElementType())) {
                case CHAR:
                    return mdDeviceArray.getNativeView().getByte(flatIndex);
                case SINT16:
                    return mdDeviceArray.getNativeView().getShort(flatIndex);
                case SINT32:
                    return mdDeviceArray.getNativeView().getInt(flatIndex);
                case SINT64:
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
        if (this.canSkipScheduling()) {
            // Fast path, skip the DAG scheduling;
            writeArrayElementImpl(index, value, valueLibrary, elementTypeProfile);
        } else {
            new MultiDimDeviceArrayViewWriteExecution(this, index, value, valueLibrary, elementTypeProfile).schedule();
        }
    }

    public void writeArrayElementImpl(long index, Object value,
                                      InteropLibrary valueLibrary,
                                      ValueProfile elementTypeProfile) throws UnsupportedTypeException {
        if ((thisDimension + 1) == mdDeviceArray.getNumberDimensions()) {
            long flatIndex = offset + index * stride;
            try {
                switch (elementTypeProfile.profile(mdDeviceArray.getElementType())) {
                    case CHAR:
                        mdDeviceArray.getNativeView().setByte(flatIndex, valueLibrary.asByte(value));
                        break;
                    case SINT16:
                        mdDeviceArray.getNativeView().setShort(flatIndex, valueLibrary.asShort(value));
                        break;
                    case SINT32:
                        mdDeviceArray.getNativeView().setInt(flatIndex, valueLibrary.asInt(value));
                        break;
                    case SINT64:
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

    public MultiDimDeviceArray getMdDeviceArray() {
        return mdDeviceArray;
    }
}
