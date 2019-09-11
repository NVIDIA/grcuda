/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.LittleEndianNativeArrayView;
import com.oracle.truffle.api.interop.ForeignAccess;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.nodes.Node;
import com.oracle.truffle.api.profiles.ValueProfile;

public final class DeviceArray implements TruffleObject {

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

    public boolean isIndexValid(long index) {
        return (index >= 0) && (index < numElements);
    }

    @Override
    public ForeignAccess getForeignAccess() {
        return DeviceArrayForeign.ACCESS;
    }

    long getSizeElements() {
        return numElements;
    }

    Number readElement(long index) {
        if ((index < 0) || (index >= numElements)) {
            throw new ArrayIndexOutOfBoundsException();
        }
        switch (elementType) {
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

    void writeElement(long index, Number value) {
        if ((index < 0) || (index >= numElements)) {
            throw new ArrayIndexOutOfBoundsException();
        }
        switch (elementType) {
            case BYTE:
            case CHAR:
                nativeView.setByte(index, value.byteValue());
                break;
            case SHORT:
                nativeView.setShort(index, value.shortValue());
                break;
            case INT:
                nativeView.setInt(index, value.intValue());
                break;
            case LONG:
                nativeView.setLong(index, value.longValue());
                break;
            case FLOAT:
                nativeView.setFloat(index, value.floatValue());
                break;
            case DOUBLE:
                nativeView.setDouble(index, value.doubleValue());
                break;
        }
    }

    @Override
    public String toString() {
        return "DeviceArray(elementType=" + elementType +
                        ", numElements=" + numElements + ", nativeView=" + nativeView + ')';
    }

    @Override
    protected void finalize() throws Throwable {
        runtime.cudaFree(nativeView);
        super.finalize();
    }

    public static final class ReadElementNode extends Node {
        private final ValueProfile profile = ValueProfile.createIdentityProfile();

        public Number readElement(DeviceArray deviceArray, long index) {
            if ((index < 0) || (index >= deviceArray.numElements)) {
                throw new ArrayIndexOutOfBoundsException();
            }
            switch (profile.profile(deviceArray.elementType)) {
                case BYTE:
                case CHAR:
                    return deviceArray.nativeView.getByte(index);
                case SHORT:
                    return deviceArray.nativeView.getShort(index);
                case INT:
                    return deviceArray.nativeView.getInt(index);
                case LONG:
                    return deviceArray.nativeView.getLong(index);
                case FLOAT:
                    return deviceArray.nativeView.getFloat(index);
                case DOUBLE:
                    return deviceArray.nativeView.getDouble(index);
            }
            return null;
        }
    }

    public static final class WriteElementNode extends Node {
        private final ValueProfile profile = ValueProfile.createIdentityProfile();

        public void writeElement(DeviceArray deviceArray, long index, Number value) {
            if ((index < 0) || (index >= deviceArray.numElements)) {
                throw new ArrayIndexOutOfBoundsException();
            }
            switch (profile.profile(deviceArray.elementType)) {
                case BYTE:
                case CHAR:
                    deviceArray.nativeView.setByte(index, value.byteValue());
                    break;
                case SHORT:
                    deviceArray.nativeView.setShort(index, value.shortValue());
                    break;
                case INT:
                    deviceArray.nativeView.setInt(index, value.intValue());
                    break;
                case LONG:
                    deviceArray.nativeView.setLong(index, value.longValue());
                    break;
                case FLOAT:
                    deviceArray.nativeView.setFloat(index, value.floatValue());
                    break;
                case DOUBLE:
                    deviceArray.nativeView.setDouble(index, value.doubleValue());
                    break;
            }
        }
    }
}
