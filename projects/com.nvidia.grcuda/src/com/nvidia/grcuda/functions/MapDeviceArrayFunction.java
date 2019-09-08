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
package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.DeviceArray;
import com.nvidia.grcuda.ElementType;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.GenerateUncached;
import com.oracle.truffle.api.dsl.Specialization;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.nodes.Node;
import com.oracle.truffle.api.profiles.LoopConditionProfile;
import com.oracle.truffle.api.profiles.ValueProfile;

@GenerateUncached
abstract class MapArrayNode extends Node {

    abstract Object execute(Object source, ElementType elementType, CUDARuntime runtime);

    @Specialization(limit = "3")
    Object doMap(Object source, ElementType elementType, CUDARuntime runtime,
                    @CachedLibrary("source") InteropLibrary interop,
                    @CachedLibrary(limit = "3") InteropLibrary elementInterop,
                    @Cached("createCountingProfile()") LoopConditionProfile loopProfile,
                    @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) {

        if (source instanceof DeviceArray) {
            return source;
        }

        if (!interop.hasArrayElements(source)) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("cannot map from non-array to DeviceArray");
        }

        long size;
        try {
            size = interop.getArraySize(source);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("cannot read array size");
        }
        loopProfile.profileCounted(size);
        DeviceArray result = new DeviceArray(runtime, size, elementType);

        for (int i = 0; loopProfile.inject(i < size); i++) {
            Object element;
            try {
                element = interop.readArrayElement(source, i);
            } catch (UnsupportedMessageException | InvalidArrayIndexException e) {
                CompilerDirectives.transferToInterpreter();
                throw new GrCUDAException("cannot read array element " + i);
            }
            try {
                result.writeArrayElement(i, element, elementInterop, elementTypeProfile);
            } catch (UnsupportedTypeException | InvalidArrayIndexException e) {
                CompilerDirectives.transferToInterpreter();
                throw new GrCUDAException("cannot coerce array element at index " + i + " to " + elementType);
            }
        }

        return result;
    }
}

@ExportLibrary(InteropLibrary.class)
public final class MapDeviceArrayFunction extends Function {

    private final CUDARuntime runtime;

    public MapDeviceArrayFunction(CUDARuntime runtime) {
        super("MapDeviceArray", "");
        this.runtime = runtime;
    }

    @ExportMessage
    public Object execute(Object[] arguments,
                    @Cached("createIdentityProfile()") ValueProfile elementTypeStringProfile,
                    @Cached("createIdentityProfile()") ValueProfile elementTypeProfile,
                    @Cached MapArrayNode mapNode) throws ArityException, UnsupportedTypeException {
        if (arguments.length < 1) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(1, arguments.length);
        }
        String typeName = elementTypeStringProfile.profile(expectString(arguments[0], "first argument of DeviceArray must be string (type name)"));
        ElementType elementType;
        try {
            elementType = elementTypeProfile.profile(ElementType.lookupType(typeName));
        } catch (TypeException e) {
            CompilerDirectives.transferToInterpreter();
            throw new RuntimeException(e.getMessage());
        }
        if (arguments.length == 1) {
            return new TypedMapDeviceArrayFunction(runtime, elementType);
        } else {
            if (arguments.length != 2) {
                CompilerDirectives.transferToInterpreter();
                throw ArityException.create(2, arguments.length);
            }
            return mapNode.execute(arguments[1], elementType, runtime);
        }
    }
}
