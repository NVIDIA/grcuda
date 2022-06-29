/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
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
package com.nvidia.grcuda.runtime;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public class DeviceList implements TruffleObject, Iterable<Device> {

    protected final List<Device> devices;

    public DeviceList(CUDARuntime runtime) {
        this(runtime.getNumberOfAvailableGPUs(), runtime);
    }

    public DeviceList(int numDevices, CUDARuntime runtime) {
        devices = Arrays.asList(new Device[numDevices]);
        this.initializeDeviceList(numDevices, runtime);
    }

    public void initializeDeviceList(int numDevices, CUDARuntime runtime) {
        for (int deviceOrdinal = 0; deviceOrdinal < numDevices; deviceOrdinal++) {
            devices.set(deviceOrdinal, new Device(deviceOrdinal, runtime));
        }
    }

    public List<Device> getDevices() {
        return this.devices;
    }

    // Java API

    public Iterator<Device> iterator() {
        return new Iterator<Device>() {
            int nextIndex = 0;

            public boolean hasNext() {
                return nextIndex < devices.size();
            }

            public Device next() {
                if (nextIndex < devices.size()) {
                    return devices.get(nextIndex++);
                } else {
                    CompilerDirectives.transferToInterpreter();
                    throw new NoSuchElementException();
                }
            }
        };
    }

    public int size() {
        return devices.size();
    }

    public Device getDevice(int deviceOrdinal) {
        if ((deviceOrdinal < 0) || (deviceOrdinal >= devices.size())) {
            CompilerDirectives.transferToInterpreter();
            throw new IndexOutOfBoundsException();
        }
        return devices.get(deviceOrdinal);
    }

    /**
     * Cleanup and deallocate the streams managed by each device;
     */
    public void cleanup() {
        this.devices.forEach(Device::cleanup);
    }

    @Override
    public String toString() {
        boolean notFirst = false;
        StringBuffer buf = new StringBuffer("[");
        for (Device device : devices) {
            if (notFirst) {
                buf.append(", ");
            }
            buf.append(device.toString());
            notFirst = true;
        }
        buf.append(']');
        return buf.toString();
    }

    // Implementation of Truffle API

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    public long getArraySize() {
        return devices.size();
    }

    @ExportMessage
    boolean isArrayElementReadable(long index) {
        return index >= 0 && index < devices.size();
    }

    @ExportMessage
    Object readArrayElement(long index) throws InvalidArrayIndexException {
        if ((index < 0) || (index >= devices.size())) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        return devices.get((int) index);
    }
}
