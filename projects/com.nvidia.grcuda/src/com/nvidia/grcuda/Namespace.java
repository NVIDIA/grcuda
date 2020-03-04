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

import java.util.Optional;
import java.util.TreeMap;

import com.nvidia.grcuda.DeviceArray.MemberSet;
import com.nvidia.grcuda.functions.Function;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

/**
 * A namespace that exposes a simple interface via {@link InteropLibrary}. It is immutable from the
 * point of view of the {@link InteropLibrary interop} API.
 */
@ExportLibrary(InteropLibrary.class)
public final class Namespace implements TruffleObject {

    private final TreeMap<String, Object> map = new TreeMap<>();

    private final String name;

    public Namespace(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name == null ? "<root>" : name;
    }

    private void addInternal(String newName, Object newElement) {
        if (newName == null || newName.isEmpty()) {
            throw new GrCUDAInternalException("cannot add elmenelement with name '" + newName + "' in namespace '" + name + "'");
        }
        if (map.containsKey(newName)) {
            throw new GrCUDAInternalException("'" + newName + "' already exists in namespace '" + name + "'");
        }
        map.put(newName, newElement);
    }

    public void addFunction(Function function) {
        addInternal(function.getName(), function);
    }

    public void addNamespace(Namespace namespace) {
        addInternal(namespace.name, namespace);
    }

    @TruffleBoundary
    public Optional<Object> lookup(String... path) {
        if (path.length == 0) {
            return Optional.empty();
        }
        return lookup(0, path);
    }

    private Optional<Object> lookup(int pos, String[] path) {
        Object entry = map.get(path[pos]);
        if (entry == null) {
            return Optional.empty();
        }
        if (pos + 1 == path.length) {
            return Optional.of(entry);
        } else {
            return entry instanceof Namespace ? ((Namespace) entry).lookup(pos + 1, path) : Optional.empty();
        }
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    public boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @TruffleBoundary
    public Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return new MemberSet(map.keySet().toArray(new String[0]));
    }

    @ExportMessage
    @TruffleBoundary
    public boolean isMemberReadable(String member) {
        return map.containsKey(member);
    }

    @ExportMessage
    @TruffleBoundary
    public Object readMember(String member) throws UnknownIdentifierException {
        Object entry = map.get(member);
        if (entry == null) {
            throw UnknownIdentifierException.create(member);
        }
        return entry;
    }

    @ExportMessage
    @TruffleBoundary
    public boolean isMemberInvocable(String member) {
        return map.get(member) instanceof Function;
    }

    @ExportMessage
    public Object invokeMember(String member, Object[] arguments,
                    @CachedLibrary(limit = "2") InteropLibrary callLibrary)
                    throws UnsupportedMessageException, ArityException, UnknownIdentifierException, UnsupportedTypeException {
        return callLibrary.execute(readMember(member), arguments);
    }
}
