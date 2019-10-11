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
package com.nvidia.grcuda.gpu;

import java.util.EnumSet;
import java.util.HashMap;

import com.nvidia.grcuda.gpu.CUDARuntime.CUDADeviceAttribute;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.CompilationFinal;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public final class GPUDeviceProperties implements TruffleObject {

    private final CUDARuntime runtime;
    private final int deviceId;

    private static final PropertySet PROPERTY_SET = new PropertySet();

    private final HashMap<String, Object> properties = new HashMap<>();

    public GPUDeviceProperties(int deviceId, CUDARuntime runtime) {
        this.deviceId = deviceId;
        this.runtime = runtime;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings({"static-method", "unused"})
    Object getMembers(boolean includeInternal) {
        return PROPERTY_SET;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String member) {
        return PROPERTY_SET.contains(member);
    }

    @ExportMessage
    Object readMember(String member) throws UnknownIdentifierException {
        if (!isMemberReadable(member)) {
            CompilerDirectives.transferToInterpreter();
            throw UnknownIdentifierException.create(member);
        }
        Object value = properties.get(member);
        if (value == null) {
            DeviceProperty prop = PROPERTY_SET.getProperty(member);
            value = prop.getValue(deviceId, runtime);
            properties.put(member, value);
        }
        return value;
    }

    @Override
    public String toString() {
        return "GPUDeviceProperties(deviceId=" + deviceId + ")";
    }

    @ExportLibrary(InteropLibrary.class)
    public static final class PropertySet implements TruffleObject {

        @CompilationFinal private final HashMap<String, DeviceProperty> propertyMap = new HashMap<>();
        @CompilationFinal(dimensions = 1) private final String[] names;

        PropertySet() {
            // Add 'CUDA device attributes' properties
            EnumSet.allOf(CUDADeviceAttribute.class).forEach(attr -> {
                propertyMap.put(attr.getAttributeName(), new DeviceAttributeProperty(attr));
            });
            String[] propertyNames = new String[propertyMap.size()];
            this.names = propertyMap.keySet().toArray(propertyNames);
        }

        private boolean contains(String name) {
            return propertyMap.containsKey(name);
        }

        private DeviceProperty getProperty(String name) {
            return propertyMap.get(name);
        }

        @ExportMessage
        @SuppressWarnings("static-method")
        public boolean hasArrayElements() {
            return true;
        }

        @ExportMessage
        public long getArraySize() {
            return names.length;
        }

        @ExportMessage
        public boolean isArrayElementReadable(long index) {
            return index >= 0 && index < propertyMap.size();
        }

        @ExportMessage
        public Object readArrayElement(long index) throws InvalidArrayIndexException {
            if ((index < 0) || (index >= propertyMap.size())) {
                CompilerDirectives.transferToInterpreter();
                throw InvalidArrayIndexException.create(index);
            }
            return names[(int) index];
        }
    }

    private interface DeviceProperty {

        String getName();

        Object getValue(int deviceId, CUDARuntime runtime);

    }

    private static class DeviceAttributeProperty implements DeviceProperty {

        private final CUDADeviceAttribute attribute;

        DeviceAttributeProperty(CUDADeviceAttribute attribute) {
            this.attribute = attribute;
        }

        public String getName() {
            return getName();
        }

        public Object getValue(int deviceId, CUDARuntime runtime) {
            return runtime.cudaDeviceGetAttribute(attribute, deviceId);
        }
    }
}
