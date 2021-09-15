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
package com.nvidia.grcuda.functions;

import java.util.ArrayList;
import java.util.Optional;

import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.MemberSet;
import com.nvidia.grcuda.runtime.array.MultiDimDeviceArray;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Cached.Shared;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

@ExportLibrary(InteropLibrary.class)
public final class DeviceArrayFunction extends Function {

    private static final String MAP = "map";

    private static final MemberSet MEMBERS = new MemberSet(MAP);

    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;

    public DeviceArrayFunction(AbstractGrCUDAExecutionContext grCUDAExecutionContext) {
        super("DeviceArray");
        this.grCUDAExecutionContext = grCUDAExecutionContext;
    }

    @Override
    @TruffleBoundary
    public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
        if (arguments.length < 1) {
            throw ArityException.create(1, arguments.length);
        }
        String typeName = expectString(arguments[0], "first argument of DeviceArray must be string (type name)");
        Type elementType;
        try {
            elementType = Type.fromGrCUDATypeString(typeName);
        } catch (TypeException e) {
            throw new GrCUDAException(e.getMessage());
        }
        if (arguments.length == 1) {
            return new TypedDeviceArrayFunction(grCUDAExecutionContext, elementType);
        } else {
            return createArray(arguments, 1, elementType, grCUDAExecutionContext);
        }
    }

    static Object createArray(Object[] arguments, int start, Type elementType, AbstractGrCUDAExecutionContext grCUDAExecutionContext) throws UnsupportedTypeException {
        ArrayList<Long> elementsPerDim = new ArrayList<>();
        Optional<Boolean> useColumnMajor = Optional.empty();
        for (int i = start; i < arguments.length; ++i) {
            Object arg = arguments[i];
            if (INTEROP.isString(arg)) {
                if (useColumnMajor.isPresent()) {
                    throw new GrCUDAException("string option already provided");
                } else {
                    String strArg = expectString(arg, "string argument expected that specifies order ('C' or 'F')");
                    if (strArg.equals("f") || strArg.equals("F")) {
                        useColumnMajor = Optional.of(true);
                    } else if (strArg.equals("c") || strArg.equals("C")) {
                        useColumnMajor = Optional.of(false);
                    } else {
                        throw new GrCUDAException("invalid string argument '" + strArg + "', only \"C\" or \"F\" are allowed");
                    }
                }
            } else {
                long n = expectLong(arg, "expected number argument for dimension size");
                if (n < 1) {
                    throw new GrCUDAException("array dimension less than 1");
                }
                elementsPerDim.add(n);
            }
        }
        if (elementsPerDim.size() == 1) {
            return new DeviceArray(grCUDAExecutionContext, elementsPerDim.get(0), elementType);
        }
        long[] dimensions = elementsPerDim.stream().mapToLong(l -> l).toArray();
        return new MultiDimDeviceArray(grCUDAExecutionContext, elementType, dimensions, useColumnMajor.orElse(false));
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return MEMBERS;
    }

    @ExportMessage(name = "isMemberReadable")
    @ExportMessage(name = "isMemberInvocable")
    @SuppressWarnings("static-method")
    boolean isMemberExisting(String memberName,
                    @Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        String name = memberProfile.profile(memberName);
        return MAP.equals(name);
    }

    @ExportMessage
    Object readMember(String memberName,
                    @Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) throws UnknownIdentifierException {
        if (MAP.equals(memberProfile.profile(memberName))) {
            return new MapDeviceArrayFunction(grCUDAExecutionContext);
        }
        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(memberName);
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
