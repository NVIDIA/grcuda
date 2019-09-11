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
package com.nvidia.grcuda.gpu;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import com.nvidia.grcuda.DeviceArray;
import com.nvidia.grcuda.gpu.UnsafeHelper.MemoryObject;
import com.oracle.truffle.api.interop.ForeignAccess;
import com.oracle.truffle.api.interop.Message;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.nodes.Node;

public final class Kernel implements TruffleObject {

    private final CUDARuntime cudaRuntime;
    private final String kernelName;
    private final CUDARuntime.CUModule kernelModule;
    private final long kernelFunction;
    private final String kernelSignature;
    private int launchCount = 0;
    private ArgumentType[] argumentTypes;
    private String ptxCode;

    public Kernel(CUDARuntime cudaRuntime, String kernelName, CUDARuntime.CUModule kernelModule, long kernelFunction, String kernelSignature) {
        this.cudaRuntime = cudaRuntime;
        this.kernelName = kernelName;
        this.kernelModule = kernelModule;
        this.kernelFunction = kernelFunction;
        this.kernelSignature = kernelSignature;
        this.argumentTypes = parseSignature(kernelSignature);
    }

    public Kernel(CUDARuntime cudaRuntime, String kernelName, CUDARuntime.CUModule kernelModule, long kernelFunction,
                    String kernelSignature, String ptx) {
        this.cudaRuntime = cudaRuntime;
        this.kernelName = kernelName;
        this.kernelModule = kernelModule;
        this.kernelFunction = kernelFunction;
        this.kernelSignature = kernelSignature;
        this.argumentTypes = parseSignature(kernelSignature);
        this.ptxCode = ptx;
    }

    @Override
    public ForeignAccess getForeignAccess() {
        return KernelForeign.ACCESS;
    }

    public void launch(KernelConfig config, Object[] kernelArgs) {
        launchCount++;
        try (KernelArguments args = createKernelArguments(kernelArgs)) {
            cudaRuntime.cuLaunchKernel(this, config, args);
        }
    }

    private Node isBoxed = Message.IS_BOXED.createNode();
    private Node unbox = Message.UNBOX.createNode();

    private Number getNumber(Object valueObj, int argIdx) {
        // This special case is necessary for R. In R every value is an array, i.e., scalars
        // are arrays of length on. In such cases, the underlying value can be obtained
        // by unboxing. This methods tries to unbox a TruffleObject.
        if (valueObj instanceof TruffleObject) {
            TruffleObject truffleValue = (TruffleObject) valueObj;
            // Check if value is a boxed type and if so unbox it.
            // This is necessary for R, since scalars in R are arrays of length 1.
            if (ForeignAccess.sendIsBoxed(isBoxed, truffleValue)) {
                try {
                    valueObj = ForeignAccess.sendUnbox(unbox, truffleValue);
                } catch (UnsupportedMessageException ex) {
                    throw new RuntimeException("UNBOX message not supported on type " + truffleValue);
                }
            }
        }
        if (!(valueObj instanceof Number)) {
            throw new RuntimeException("argument " + (argIdx + 1) + " expected Number type, but is " + valueObj.getClass().getName());
        }
        return ((Number) valueObj);
    }

    private KernelArguments createKernelArguments(Object[] args) {
        if (args.length != argumentTypes.length) {
            throw new IllegalArgumentException("expected " +
                            argumentTypes.length + " kernel arguments, got " + args.length);
        }
        KernelArguments kernelArgs = new KernelArguments(args.length);
        int argIdx = 0;
        for (ArgumentType type : argumentTypes) {
            Number number;
            switch (type) {
                case INT32:
                    number = getNumber(args[argIdx], argIdx);
                    UnsafeHelper.Integer32Object int32 = UnsafeHelper.createInteger32Object();
                    int32.setValue(number.intValue());
                    kernelArgs.setArgument(argIdx, int32);
                    break;
                case INT64:
                    number = getNumber(args[argIdx], argIdx);
                    UnsafeHelper.Integer64Object int64 = UnsafeHelper.createInteger64Object();
                    int64.setValue(number.longValue());
                    kernelArgs.setArgument(argIdx, int64);
                    break;
                case FLOAT32:
                    number = getNumber(args[argIdx], argIdx);
                    UnsafeHelper.Float32Object fp32 = UnsafeHelper.createFloat32Object();
                    fp32.setValue(number.floatValue());
                    kernelArgs.setArgument(argIdx, fp32);
                    break;
                case FLOAT64:
                    number = getNumber(args[argIdx], argIdx);
                    UnsafeHelper.Float64Object fp64 = UnsafeHelper.createFloat64Object();
                    fp64.setValue(number.doubleValue());
                    kernelArgs.setArgument(argIdx, fp64);
                    break;
                case POINTER:
                    if (args[argIdx] instanceof DeviceArray) {
                        DeviceArray deviceArray = (DeviceArray) args[argIdx];
                        UnsafeHelper.PointerObject pointer = UnsafeHelper.createPointerObject();
                        pointer.setValueOfPointer(deviceArray.getPointer());
                        kernelArgs.setArgument(argIdx, pointer);
                    } else {
                        throw new IllegalArgumentException("argument " + (argIdx + 1) +
                                        " expected DeviceArray type, " + args[argIdx].getClass().getName());
                    }
                    break;
            }
            argIdx += 1;
        }
        return kernelArgs;
    }

    private static ArgumentType[] parseSignature(String kernelSignature) {
        ArrayList<ArgumentType> args = new ArrayList<>();
        for (String s : kernelSignature.trim().split(",")) {
            ArgumentType type;
            switch (s.trim()) {
                case "pointer":
                    type = ArgumentType.POINTER;
                    break;
                case "uint64":
                case "sint64":
                    type = ArgumentType.INT64;
                    break;
                case "uint32":
                case "sint32":
                    type = ArgumentType.INT32;
                    break;
                case "float":
                    type = ArgumentType.FLOAT32;
                    break;
                case "double":
                    type = ArgumentType.FLOAT64;
                    break;
                default:
                    throw new IllegalArgumentException("invalid type identifier in kernel signature: " + s);
            }
            args.add(type);
        }
        ArgumentType[] argArray = new ArgumentType[args.size()];
        args.toArray(argArray);
        return argArray;
    }

    public long getKernelFunction() {
        return kernelFunction;
    }

    public void dispose() {
        kernelModule.decrementRefCount();
    }

    @Override
    public String toString() {
        return "Kernel(" + kernelName + ", " + kernelSignature + ", launchCount=" + launchCount + ")";
    }

    private enum ArgumentType {
        POINTER,
        INT32,
        INT64,
        FLOAT32,
        FLOAT64;
    }

    public String getPTX() {
        return ptxCode;
    }

    public String getKernelName() {
        return kernelName;
    }

    public int getLaunchCount() {
        return launchCount;
    }
}

final class KernelArguments implements Closeable {

    private final UnsafeHelper.PointerArray argumentArray;
    private final ArrayList<Closeable> argumentValues = new ArrayList<>();

    KernelArguments(int numArgs) {
        this.argumentArray = UnsafeHelper.createPointerArray(numArgs);
    }

    public void setArgument(int argIdx, MemoryObject obj) {
        argumentArray.setValueAt(argIdx, obj.getAddress());
        argumentValues.add(obj);
    }

    long getPointer() {
        return argumentArray.getAddress();
    }

    @Override
    public void close() {
        this.argumentArray.close();
        for (Closeable c : argumentValues) {
            try {
                c.close();
            } catch (IOException e) {
                /* ignored */
            }
        }
    }
}
