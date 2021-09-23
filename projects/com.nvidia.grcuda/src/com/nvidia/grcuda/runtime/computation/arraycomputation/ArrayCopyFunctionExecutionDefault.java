/*
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
package com.nvidia.grcuda.runtime.computation.arraycomputation;

import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.profiles.ValueProfile;

/**
 * Slow-path implementation of the copy function on {@link AbstractArray}, it copies data using a simple loop.
 * It is not as fast as a memcpy, but it avoids some overheads of doing the copy in the host language;
 */
public class ArrayCopyFunctionExecutionDefault extends ArrayCopyFunctionExecution {
    /**
     * InteropLibrary object used to access the other array's elements;
     */
    private final InteropLibrary pointerAccess;
    /**
     * Object that identifies the other array from/to which we copy data;
     */
    private final Object otherArray;

    public ArrayCopyFunctionExecutionDefault(AbstractArray array, DeviceArrayCopyFunction.CopyDirection direction, long numElements,
                                             @CachedLibrary(limit = "3") InteropLibrary pointerAccess,
                                             Object otherArray, ArrayCopyFunctionExecutionInitializer dependencyInitializer) {
        super(array, direction, numElements, dependencyInitializer);
        this.pointerAccess = pointerAccess;
        this.otherArray = otherArray;
    }

    @Override
    void executeInner() {
        ValueProfile elementTypeProfile = ValueProfile.createIdentityProfile();
        try {
            if (direction == DeviceArrayCopyFunction.CopyDirection.FROM_POINTER) {
                InteropLibrary valueLibrary = InteropLibrary.getFactory().createDispatched(5);
                for (long i = 0; i < this.numElements; i++) {
                    this.array.writeNativeView(i, this.pointerAccess.readArrayElement(this.otherArray, i), valueLibrary, elementTypeProfile);
                }
            } else if (direction == DeviceArrayCopyFunction.CopyDirection.TO_POINTER) {
                for (long i = 0; i < this.numElements; i++) {
                    this.pointerAccess.writeArrayElement(this.otherArray, i, this.array.readNativeView(i, elementTypeProfile));
                }
            } else {
                CompilerDirectives.transferToInterpreter();
                throw new DeviceArrayCopyException("invalid direction for copy: " + direction);
            }
        } catch (InvalidArrayIndexException | UnsupportedMessageException | UnsupportedTypeException e) {
            throw new DeviceArrayCopyException("invalid array copy: " + e);
        }
    }

    @Override
    public String toString() {
        try {
            return "array  copy on " + System.identityHashCode(array) + "; direction=" + direction + "; target=" + pointerAccess.asString(this.otherArray) + "; size=" + numElements;
        } catch (UnsupportedMessageException e) {
            return "array copy on " + System.identityHashCode(array) + "; direction=" + direction + "; size=" + numElements;
        }
    }
}
