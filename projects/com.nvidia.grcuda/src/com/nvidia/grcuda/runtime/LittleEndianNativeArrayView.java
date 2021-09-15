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
package com.nvidia.grcuda.runtime;

import com.nvidia.grcuda.Type;
import sun.misc.Unsafe;

/**
 * A non-owning view over native memory provided. No bounds checks are performed.
 */
public class LittleEndianNativeArrayView {

    private static final Unsafe unsafe = UnsafeHelper.getUnsafe();
    private final long startAddress;
    private final long sizeInBytes;

    public void setByte(long index, byte value) {
        unsafe.putByte(startAddress + index * Type.CHAR.getSizeBytes(), value);
    }

    public void setChar(long index, char value) {
        unsafe.putChar(startAddress + index * Type.SINT16.getSizeBytes(), value);
    }

    public void setShort(long index, short value) {
        unsafe.putShort(startAddress + index * Type.SINT16.getSizeBytes(), value);
    }

    public void setInt(long index, int value) {
        unsafe.putInt(startAddress + index * Type.SINT32.getSizeBytes(), value);
    }

    public void setLong(long index, long value) {
        unsafe.putLong(startAddress + index * Type.SINT64.getSizeBytes(), value);
    }

    public void setFloat(long index, float value) {
        unsafe.putFloat(startAddress + index * Type.FLOAT.getSizeBytes(), value);
    }

    public void setDouble(long index, double value) {
        unsafe.putDouble(startAddress + index * Type.DOUBLE.getSizeBytes(), value);
    }

    public byte getByte(long index) {
        return unsafe.getByte(startAddress + index * Type.CHAR.getSizeBytes());
    }

    public char getChar(long index) {
        return unsafe.getChar(startAddress + index * Type.CHAR.getSizeBytes());
    }

    public short getShort(long index) {
        return unsafe.getShort(startAddress + index * Type.SINT32.getSizeBytes());
    }

    public int getInt(long index) {
        return unsafe.getInt(startAddress + index * Type.SINT32.getSizeBytes());
    }

    public long getLong(long index) {
        return unsafe.getLong(startAddress + index * Type.SINT64.getSizeBytes());
    }

    public float getFloat(long index) {
        return unsafe.getFloat(startAddress + index * Type.FLOAT.getSizeBytes());
    }

    public double getDouble(long index) {
        return unsafe.getDouble(startAddress + index * Type.DOUBLE.getSizeBytes());
    }

    public long getStartAddress() {
        return startAddress;
    }

    public long getSizeInBytes() {
        return sizeInBytes;
    }

    @Override
    public String toString() {
        return String.format("LittleEndianNativearrayView(startAddress=0x%016x, sizeInBytes=%d)",
                        startAddress, sizeInBytes);
    }

    LittleEndianNativeArrayView(long startAddress, long sizeInBytes) {
        this.startAddress = startAddress;
        this.sizeInBytes = sizeInBytes;
    }
}
