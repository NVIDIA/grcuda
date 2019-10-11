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
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Cached.Shared;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

@ExportLibrary(InteropLibrary.class)
public final class Device implements TruffleObject {

    private static final String ID = "id";
    private static final String PROPERTIES = "properties";
    private static final MemberSet PUBLIC_MEMBERS = new MemberSet(ID, PROPERTIES);

    private final int deviceId;
    private final GPUDeviceProperties properties;

    public Device(int deviceId, CUDARuntime runtime) {
        this.deviceId = deviceId;
        this.properties = new GPUDeviceProperties(deviceId, runtime);
    }

    public GPUDeviceProperties getProperties() {
        return properties;
    }

    @Override
    public String toString() {
        return "Device(id=" + deviceId + ")";
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
    boolean isMemberReadable(String member,
                    @Shared("member") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        return ID.equals(memberProfile.profile(member)) || PROPERTIES.equals(memberProfile.profile(member));
    }

    @ExportMessage
    Object readMember(String member,
                    @Shared("member") @Cached("createIdentityProfile()") ValueProfile memberProfile) throws UnknownIdentifierException {
        if (!isMemberReadable(member, memberProfile)) {
            CompilerDirectives.transferToInterpreter();
            throw UnknownIdentifierException.create(member);
        }
        if (ID.equals(member)) {
            return deviceId;
        }
        if (PROPERTIES.equals(member)) {
            return properties;
        }
        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(member);
    }
}
