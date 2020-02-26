/*
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
package com.nvidia.grcuda.functions.map;

import static com.nvidia.grcuda.functions.map.MapFunction.checkArity;

import com.nvidia.grcuda.DeviceArray.MemberSet;
import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives;
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

@ExportLibrary(InteropLibrary.class)
public final class MapArgObject implements TruffleObject {

    // This returns only a certain set of members, although any member can be read.
    protected static final String BIND = "bind";
    protected static final String MAP = "map";
    protected static final String SHRED = "shred";
    protected static final String DESCRIBE = "describe";
    private static final MemberSet MEMBERS = new MemberSet(BIND, MAP, SHRED, DESCRIBE);

    final MapArgObjectBase value;

    public MapArgObject(MapArgObjectBase value) {
        this.value = value;
    }

    // Java API

    public MapArgObject map(Object function) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        if (!MapFunction.INTEROP.isExecutable(function)) {
            throw UnsupportedTypeException.create(new Object[]{function}, "expecting executable mapping function");
        }
        return new MapArgObject(new MapArgObjectMap(function, value));
    }

    public MapArgObject shred() {
        CompilerAsserts.neverPartOfCompilation();
        return new MapArgObject(new MapArgObjectShred(value));
    }

    public String describe() {
        CompilerAsserts.neverPartOfCompilation();
        return value.toString();
    }

    // Interop API

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        // This returns a fixed set, although any member can be read.
        return MEMBERS;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberReadable(@SuppressWarnings("unused") String member) {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("unused")
    public MapArgObject readMember(String member) {
        return new MapArgObject(new MapArgObjectMember(member, value));
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    boolean isMemberInvocable(String member) {
        return MEMBERS.constainsValue(member);
    }

    @ExportMessage
    Object invokeMember(String member, Object[] arguments,
                    @CachedLibrary("this") InteropLibrary interopRead,
                    @CachedLibrary(limit = "1") InteropLibrary interopExecute) throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        return interopExecute.execute(interopRead.readMember(this, member), arguments);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    long getArraySize() {
        // This returns a length of 0, although any element can be read.
        return 0;
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    boolean isArrayElementReadable(@SuppressWarnings("unused") long index) {
        return true;
    }

    @ExportMessage
    public Object readArrayElement(long index) {
        return new MapArgObject(new MapArgObjectElement(index, value));
    }

    @ExportMessage
    boolean isExecutable() {
        // can be executed if this refers to "map", "shred", etc.
        return value instanceof MapArgObjectMember && MEMBERS.constainsValue(((MapArgObjectMember) value).name);
    }

    @ExportMessage
    @TruffleBoundary
    Object execute(Object[] arguments) throws UnsupportedMessageException, UnsupportedTypeException, ArityException {
        if (value instanceof MapArgObjectMember) {
            MapArgObjectMember member = (MapArgObjectMember) value;
            if (MAP.equals(member.name)) {
                checkArity(arguments, 1);
                return new MapArgObject(member.parent).map(arguments[0]);
            } else if (SHRED.equals(member.name)) {
                checkArity(arguments, 0);
                CompilerAsserts.neverPartOfCompilation();
                return new MapArgObject(member.parent).shred();
            } else if (DESCRIBE.equals(member.name)) {
                checkArity(arguments, 0);
                CompilerAsserts.neverPartOfCompilation();
                return new MapArgObject(member.parent).describe();
            } else if (BIND.equals(member.name)) {
                checkArity(arguments, 3);
                return member.parent.bind(arguments[0], arguments[1], arguments[2]);
            }
        }
        CompilerDirectives.transferToInterpreter();
        throw UnsupportedMessageException.create();
    }
}
