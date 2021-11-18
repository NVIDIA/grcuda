/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
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
package com.nvidia.grcuda.runtime;

import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.MultiDimDeviceArray;
import com.nvidia.grcuda.runtime.computation.KernelExecution;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public class ConfiguredKernel extends ProfilableElement implements TruffleObject {

    private final Kernel kernel;

    private final KernelConfig config;

    public ConfiguredKernel(Kernel kernel, KernelConfig config) {
        this.kernel = kernel;
        this.config = config;
    }

    @ExportMessage
    boolean isExecutable() {
        return true;
    }

    /**
     * Parse the input arguments of the kernel and map them to the signature, making sure that the signature is respected
     * @param args list of input arguments to the kernel
     * @param booleanAccess used to parse boolean inputs
     * @param int8Access used to parse char inputs
     * @param int16Access used to parse short integer inputs
     * @param int32Access used to parse integer inputs
     * @param int64Access used to parse long integer inputs
     * @param doubleAccess used to parse double and float inputs
     * @return the object that wraps the kernel signature and arguments
     * @throws UnsupportedTypeException if one of the inputs does not respect the signature
     * @throws ArityException if the number of inputs does not respect the signature
     */
    KernelArguments createKernelArguments(Object[] args, InteropLibrary booleanAccess,
                                          InteropLibrary int8Access, InteropLibrary int16Access,
                                          InteropLibrary int32Access, InteropLibrary int64Access, InteropLibrary doubleAccess)
            throws UnsupportedTypeException, ArityException {
        if (args.length != kernel.getKernelParameters().length) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(kernel.getKernelParameters().length, args.length);
        }
        KernelArguments kernelArgs = new KernelArguments(args, this.kernel.getKernelParameters());
        for (int paramIdx = 0; paramIdx < kernel.getKernelParameters().length; paramIdx++) {
            Object arg = args[paramIdx];
            ComputationArgument param = kernel.getKernelParameters()[paramIdx];
            Type paramType = param.getType();
            try {
                if (param.isPointer()) {
                    if (arg instanceof DeviceArray) {
                        DeviceArray deviceArray = (DeviceArray) arg;
                        if (!param.isSynonymousWithPointerTo(deviceArray.getElementType())) {
                            throw new GrCUDAException("device array of " + deviceArray.getElementType() + " cannot be used as pointer argument " + paramType);
                        }
                        UnsafeHelper.PointerObject pointer = UnsafeHelper.createPointerObject();
                        pointer.setValueOfPointer(deviceArray.getPointer());
                        kernelArgs.setArgument(paramIdx, pointer);
                    } else if (arg instanceof MultiDimDeviceArray) {
                        MultiDimDeviceArray deviceArray = (MultiDimDeviceArray) arg;
                        if (!param.isSynonymousWithPointerTo(deviceArray.getElementType())) {
                            throw new GrCUDAException("multi-dimensional device array of " +
                                    deviceArray.getElementType() + " cannot be used as pointer argument " + paramType);
                        }
                        UnsafeHelper.PointerObject pointer = UnsafeHelper.createPointerObject();
                        pointer.setValueOfPointer(deviceArray.getPointer());
                        kernelArgs.setArgument(paramIdx, pointer);
                    } else {
                        CompilerDirectives.transferToInterpreter();
                        throw UnsupportedTypeException.create(new Object[]{arg}, "expected DeviceArray type");
                    }
                } else {
                    // by-value argument
                    switch (paramType) {
                        case BOOLEAN: {
                            UnsafeHelper.Integer8Object int8 = UnsafeHelper.createInteger8Object();
                            int8.setValue(booleanAccess.asBoolean(arg) ? ((byte) 1) : ((byte) 0));
                            kernelArgs.setArgument(paramIdx, int8);
                            break;
                        }
                        case SINT8:
                        case CHAR: {
                            UnsafeHelper.Integer8Object int8 = UnsafeHelper.createInteger8Object();
                            int8.setValue(int8Access.asByte(arg));
                            kernelArgs.setArgument(paramIdx, int8);
                            break;
                        }
                        case SINT16: {
                            UnsafeHelper.Integer16Object int16 = UnsafeHelper.createInteger16Object();
                            int16.setValue(int16Access.asShort(arg));
                            kernelArgs.setArgument(paramIdx, int16);
                            break;
                        }
                        case SINT32:
                        case WCHAR: {
                            UnsafeHelper.Integer32Object int32 = UnsafeHelper.createInteger32Object();
                            int32.setValue(int32Access.asInt(arg));
                            kernelArgs.setArgument(paramIdx, int32);
                            break;
                        }
                        case SINT64:
                        case SLL64:
                            // no larger primitive type than long -> interpret long as unsigned
                        case UINT64:
                        case ULL64: {
                            UnsafeHelper.Integer64Object int64 = UnsafeHelper.createInteger64Object();
                            int64.setValue(int64Access.asLong(arg));
                            kernelArgs.setArgument(paramIdx, int64);
                            break;
                        }
                        case UINT8:
                        case CHAR8: {
                            int uint8 = int16Access.asShort(arg);
                            if (uint8 < 0 || uint8 > 0xff) {
                                CompilerDirectives.transferToInterpreter();
                                throw createExceptionValueOutOfRange(paramType, uint8);
                            }
                            UnsafeHelper.Integer8Object int8 = UnsafeHelper.createInteger8Object();
                            int8.setValue((byte) (0xff & uint8));
                            kernelArgs.setArgument(paramIdx, int8);
                            break;
                        }
                        case UINT16:
                        case CHAR16: {
                            int uint16 = int32Access.asInt(arg);
                            if (uint16 < 0 || uint16 > 0xffff) {
                                CompilerDirectives.transferToInterpreter();
                                throw createExceptionValueOutOfRange(paramType, uint16);
                            }
                            UnsafeHelper.Integer16Object int16 = UnsafeHelper.createInteger16Object();
                            int16.setValue((short) (0xffff & uint16));
                            kernelArgs.setArgument(paramIdx, int16);
                            break;
                        }
                        case UINT32: {
                            long uint32 = int64Access.asLong(arg);
                            if (uint32 < 0 || uint32 > 0xffffffffL) {
                                CompilerDirectives.transferToInterpreter();
                                throw createExceptionValueOutOfRange(paramType, uint32);
                            }
                            UnsafeHelper.Integer32Object int32 = UnsafeHelper.createInteger32Object();
                            int32 = UnsafeHelper.createInteger32Object();
                            int32.setValue((int) (0xffffffffL & uint32));
                            kernelArgs.setArgument(paramIdx, int32);
                            break;
                        }
                        case FLOAT: {
                            UnsafeHelper.Float32Object fp32 = UnsafeHelper.createFloat32Object();
                            // going via "double" to allow floats to be initialized with doubles
                            fp32.setValue((float) doubleAccess.asDouble(arg));
                            kernelArgs.setArgument(paramIdx, fp32);
                            break;
                        }
                        case DOUBLE: {
                            UnsafeHelper.Float64Object fp64 = UnsafeHelper.createFloat64Object();
                            fp64.setValue(doubleAccess.asDouble(arg));
                            kernelArgs.setArgument(paramIdx, fp64);
                            break;
                        }
                        default:
                            CompilerDirectives.transferToInterpreter();
                            throw UnsupportedTypeException.create(new Object[]{arg},
                                    "unsupported by-value parameter type: " + paramType);
                    }
                }
            } catch (UnsupportedMessageException e) {
                CompilerDirectives.transferToInterpreter();
                throw UnsupportedTypeException.create(new Object[]{arg},
                        "expected type " + paramType + " in argument " + arg);
            }
        }
        return kernelArgs;
    }

    private static GrCUDAException createExceptionValueOutOfRange(Type type, long value) {
        return new GrCUDAException("value " + value + " is out of range for type " + type);
    }

    @ExportMessage
    @TruffleBoundary
    Object execute(Object[] arguments,
                    @CachedLibrary(limit = "3") InteropLibrary boolAccess,
                    @CachedLibrary(limit = "3") InteropLibrary int8Access,
                    @CachedLibrary(limit = "3") InteropLibrary int16Access,
                    @CachedLibrary(limit = "3") InteropLibrary int32Access,
                    @CachedLibrary(limit = "3") InteropLibrary int64Access,
                    @CachedLibrary(limit = "3") InteropLibrary doubleAccess) throws UnsupportedTypeException, ArityException {
        kernel.incrementLaunchCount();
        try (KernelArguments args = this.createKernelArguments(arguments, boolAccess, int8Access, int16Access,
                        int32Access, int64Access, doubleAccess)) {
            // If using a manually specified stream, do not schedule it automatically, but execute it immediately;
            if (!config.useCustomStream()) {
                new KernelExecution(this, args).schedule();
            } else {
                kernel.getGrCUDAExecutionContext().getCudaRuntime().cuLaunchKernel(kernel, config, args, config.getStream());
            }
        }
        return this;
    }

    public Kernel getKernel() {
        return kernel;
    }

    public KernelConfig getConfig() {
        return config;
    }

    @Override
    public String toString() {
        return "ConfiguredKernel(" + kernel.toString() + "; " + config.toString() + ")";
    }
}
