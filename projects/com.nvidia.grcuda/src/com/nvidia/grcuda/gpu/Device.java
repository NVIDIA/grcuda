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

import com.nvidia.grcuda.DeviceArray.MemberSet;
import com.nvidia.grcuda.NoneValue;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Cached.Shared;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

@ExportLibrary(InteropLibrary.class)
public final class Device implements TruffleObject {

    private static final String ID = "id";
    private static final String PROPERTIES = "properties";
    private static final String IS_CURRENT = "isCurrent";
    private static final String SET_CURRENT = "setCurrent";
    private static final MemberSet PUBLIC_MEMBERS = new MemberSet(ID, PROPERTIES, IS_CURRENT, SET_CURRENT);

    private final int deviceId;
    private final GPUDeviceProperties properties;
    private final CUDARuntime runtime;

    public Device(int deviceId, CUDARuntime runtime) {
        this.deviceId = deviceId;
        this.runtime = runtime;
        this.properties = new GPUDeviceProperties(deviceId, runtime);
    }

    public GPUDeviceProperties getProperties() {
        return properties;
    }

    @Override
    public String toString() {
        return "Device(id=" + deviceId + ")";
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof Device) {
            Device otherDevice = (Device) other;
            return otherDevice.deviceId == deviceId;
        }
        return false;
    }

    @Override
    public int hashCode() {
        return deviceId;
    }

    // Implementation of Truffle API

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings({"static-method", "unused"})
    Object getMembers(boolean includeInternal) {
        return PUBLIC_MEMBERS;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String memberName,
                    @Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        String name = memberProfile.profile(memberName);
        return ID.equals(name) || PROPERTIES.equals(name) ||
                        IS_CURRENT.equals(name) || SET_CURRENT.equals(name);
    }

    @ExportMessage
    Object readMember(String memberName,
                    @Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) throws UnknownIdentifierException {
        if (!isMemberReadable(memberName, memberProfile)) {
            CompilerDirectives.transferToInterpreter();
            throw UnknownIdentifierException.create(memberName);
        }
        if (ID.equals(memberName)) {
            return deviceId;
        }
        if (PROPERTIES.equals(memberName)) {
            return properties;
        }
        if (IS_CURRENT.equals(memberName)) {
            return new IsCurrentFunction(deviceId, runtime);
        }
        if (SET_CURRENT.equals(memberName)) {
            return new SetCurrentFunction(deviceId, runtime);
        }
        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(memberName);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberInvocable(String memberName) {
        return IS_CURRENT.equals(memberName) || SET_CURRENT.equals(memberName);
    }

    @ExportMessage
    Object invokeMember(String memberName,
                    Object[] arguments,
                    @CachedLibrary("this") InteropLibrary interopRead,
                    @CachedLibrary(limit = "1") InteropLibrary interopExecute)
                    throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        return interopExecute.execute(interopRead.readMember(this, memberName), arguments);
    }
}

@ExportLibrary(InteropLibrary.class)
final class IsCurrentFunction implements TruffleObject {
    private final int deviceId;
    private final CUDARuntime runtime;

    IsCurrentFunction(int deviceId, CUDARuntime runtime) {
        this.deviceId = deviceId;
        this.runtime = runtime;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    public Object execute(Object[] arguments) throws ArityException {
        if (arguments.length != 0) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(0, arguments.length);
        }
        return runtime.cudaGetDevice() == deviceId;
    }
}

@ExportLibrary(InteropLibrary.class)
class SetCurrentFunction implements TruffleObject {
    private int deviceId;
    private final CUDARuntime runtime;

    SetCurrentFunction(int deviceId, CUDARuntime runtime) {
        this.deviceId = deviceId;
        this.runtime = runtime;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    public Object execute(Object[] arguments) throws ArityException {
        if (arguments.length != 0) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(0, arguments.length);
        }
        runtime.cudaSetDevice(deviceId);
        return NoneValue.get();
    }
}
