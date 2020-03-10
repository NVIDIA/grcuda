/*
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
package com.nvidia.grcuda.functions.map;

import org.graalvm.collections.EconomicMap;

import com.nvidia.grcuda.DeviceArray.MemberSet;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public final class ArgumentSet implements TruffleObject {

    final EconomicMap<String, Integer> nameList = EconomicMap.create();

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    @TruffleBoundary
    Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        String[] names = new String[nameList.size()];
        int pos = 0;
        for (String name : nameList.getKeys()) {
            names[pos++] = name;
        }
        return new MemberSet(names);
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    boolean isMemberReadable(@SuppressWarnings("unused") String member) {
        return true;
    }

    @ExportMessage
    @TruffleBoundary
    public Integer readMember(String member) {
        Integer index = nameList.get(member);
        if (index == null) {
            nameList.put(member, index = nameList.size());
        }
        return index;
    }
}
