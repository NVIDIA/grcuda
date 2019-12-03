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
package com.nvidia.grcuda.tensorrt;

import com.nvidia.grcuda.DeviceArray;
import com.nvidia.grcuda.GPUPointer;
import com.nvidia.grcuda.functions.ExternalFunction;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.gpu.UnsafeHelper;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class EnqueueFunction extends Function {

    private final ExternalFunction nfiFunction;

    protected EnqueueFunction(ExternalFunction nfiFunction) {
        super("enqueue", "TRT");
        this.nfiFunction = nfiFunction;
    }

    @Override
    protected Object call(Object[] arguments) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        checkArgumentLength(arguments, 3);
        int engineHandle = expectInt(arguments[0]);
        int batchSize = expectInt(arguments[1]);

        // extract pointers from buffers array argument
        Object bufferArg = arguments[2];
        if (!INTEROP.hasArrayElements(bufferArg)) {
            throw UnsupportedMessageException.create();
        }
        int numBuffers = (int) INTEROP.getArraySize(bufferArg);
        try (UnsafeHelper.PointerArray pointerArray = UnsafeHelper.createPointerArray(numBuffers)) {
            for (int i = 0; i < numBuffers; ++i) {
                try {
                    Object buffer = INTEROP.readArrayElement(bufferArg, i);
                    if (!(buffer instanceof DeviceArray) && !(buffer instanceof GPUPointer)) {
                        UnsupportedTypeException.create(new Object[]{buffer});
                    }
                    pointerArray.setValueAt(i, INTEROP.asPointer(buffer));
                } catch (InvalidArrayIndexException e) {
                    InvalidArrayIndexException.create(i);
                }
            }
            long stream = 0;
            long eventConsumed = 0;
            Object result = INTEROP.execute(nfiFunction, engineHandle, batchSize, pointerArray.getAddress(), stream, eventConsumed);
            if (!INTEROP.fitsInInt(result)) {
                CompilerDirectives.transferToInterpreter();
                throw new RuntimeException("result of 'enqueue' is not an int");
            }
            return INTEROP.asInt(result) == 1;
        }
    }
}
