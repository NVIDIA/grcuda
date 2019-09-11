/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import com.nvidia.grcuda.DeviceArray;
import com.nvidia.grcuda.ElementType;
import com.nvidia.grcuda.MultiDimDeviceArray;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.oracle.truffle.api.frame.VirtualFrame;

public final class DeviceArrayFunction extends Function {

    private final CUDARuntime runtime;

    public DeviceArrayFunction(CUDARuntime runtime) {
        super("DeviceArray", "");
        this.runtime = runtime;
    }

    @Override
    public Object execute(VirtualFrame frame) {
        Object[] arguments = frame.getArguments();
        if (arguments.length < 3) {   // arg 0 is the function itself
            throw new RuntimeException("DeviceArray function expects at least two arguments.");
        }
        String typeName = expectString(arguments[1], "argument 1 of DeviceArray must be string (type name)");
        ArrayList<Long> elementsPerDim = new ArrayList<>();
        Optional<Boolean> useColumnMajor = Optional.empty();
        for (int i = 2; i < arguments.length; ++i) {
            Object arg = arguments[i];
            if (isString(arg)) {
                if (useColumnMajor.isPresent()) {
                    throw new RuntimeException("string option already provided");
                } else {
                    String strArg = expectString(arg,
                                    "string argument expected that specifies order ('C' or 'F')");
                    if (strArg.equals("f") || strArg.equals("F")) {
                        useColumnMajor = Optional.of(true);
                    } else if (strArg.equals("c") || strArg.equals("C")) {
                        useColumnMajor = Optional.of(false);
                    } else {
                        throw new RuntimeException("invalid string argument '" + strArg +
                                        "', only \"C\" or \"F\" are allowed");
                    }
                }
            } else {
                Number numElements = expectNumber(arg, "expected number argument for dimension size");
                long n = numElements.longValue();
                if (n < 1) {
                    throw new RuntimeException("array dimension less than 1");
                }
                elementsPerDim.add(n);
            }
        }
        try {
            ElementType elementType = ElementType.lookupType(typeName);
            if (elementsPerDim.size() == 1) {
                return new DeviceArray(runtime, elementsPerDim.get(0), elementType);
            }
            long[] dimensions = elementsPerDim.stream().mapToLong(l -> l).toArray();
            return new MultiDimDeviceArray(runtime, elementType, dimensions, useColumnMajor.orElse(false));
        } catch (TypeException e) {
            throw new RuntimeException(e.getMessage());
        }
    }
}
