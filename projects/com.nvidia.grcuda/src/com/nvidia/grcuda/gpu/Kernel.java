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
package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.array.DeviceArray;
import com.nvidia.grcuda.array.DeviceArray.MemberSet;
import com.nvidia.grcuda.array.MultiDimDeviceArray;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Fallback;
import com.oracle.truffle.api.dsl.Specialization;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@ExportLibrary(InteropLibrary.class)
public class Kernel implements TruffleObject {

    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final String kernelName;
    private final CUDARuntime.CUModule kernelModule;
    private final long kernelFunction;
    private final String kernelSignature;
    private int launchCount = 0;
    private final ArgumentType[] argumentTypes;
    private final List<Boolean> argsAreArrays;
    private String ptxCode;

    public Kernel(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String kernelName, CUDARuntime.CUModule kernelModule, long kernelFunction, String kernelSignature) {
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.kernelName = kernelName;
        this.kernelModule = kernelModule;
        this.kernelFunction = kernelFunction;
        this.kernelSignature = kernelSignature;
        this.argumentTypes = parseSignature(kernelSignature);
        this.argsAreArrays = computeIfArgsAreArrays(this.argumentTypes);
        this.grCUDAExecutionContext.registerKernel(this);
    }

    public Kernel(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String kernelName, CUDARuntime.CUModule kernelModule, long kernelFunction,
                    String kernelSignature, String ptx) {
        this(grCUDAExecutionContext, kernelName, kernelModule, kernelFunction, kernelSignature);
        this.ptxCode = ptx;
    }

    public void incrementLaunchCount() {
        launchCount++;
    }

    public AbstractGrCUDAExecutionContext getGrCUDAExecutionContext() {
        return grCUDAExecutionContext;
    }

    public ArgumentType[] getArgumentTypes() {
        return argumentTypes;
    }

    public List<Boolean> getArgsAreArrays() {
        return argsAreArrays;
    }

    public KernelArguments createKernelArguments(Object[] args, InteropLibrary int32Access, InteropLibrary int64Access, InteropLibrary doubleAccess)
                    throws UnsupportedTypeException, ArityException {
        if (args.length != argumentTypes.length) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(argumentTypes.length, args.length);
        }
        KernelArguments kernelArgs = new KernelArguments(args);
        for (int argIdx = 0; argIdx < argumentTypes.length; argIdx++) {
            ArgumentType type = argumentTypes[argIdx];
            try {
                switch (type) {
                    case INT32:
                        UnsafeHelper.Integer32Object int32 = UnsafeHelper.createInteger32Object();
                        int32.setValue(int32Access.asInt(args[argIdx]));
                        kernelArgs.setArgument(argIdx, int32);
                        break;
                    case INT64:
                        UnsafeHelper.Integer64Object int64 = UnsafeHelper.createInteger64Object();
                        int64.setValue(int64Access.asLong(args[argIdx]));
                        kernelArgs.setArgument(argIdx, int64);
                        break;
                    case FLOAT32:
                        UnsafeHelper.Float32Object fp32 = UnsafeHelper.createFloat32Object();
                        // going via "double" to allow floats to be initialized with doubles
                        fp32.setValue((float) doubleAccess.asDouble(args[argIdx]));
                        kernelArgs.setArgument(argIdx, fp32);
                        break;
                    case FLOAT64:
                        UnsafeHelper.Float64Object fp64 = UnsafeHelper.createFloat64Object();
                        fp64.setValue(doubleAccess.asDouble(args[argIdx]));
                        kernelArgs.setArgument(argIdx, fp64);
                        break;
                    case POINTER:
                        if (args[argIdx] instanceof DeviceArray) {
                            DeviceArray deviceArray = (DeviceArray) args[argIdx];
                            UnsafeHelper.PointerObject pointer = UnsafeHelper.createPointerObject();
                            pointer.setValueOfPointer(deviceArray.getPointer());
                            kernelArgs.setArgument(argIdx, pointer);
                        } else if (args[argIdx] instanceof MultiDimDeviceArray) {
                            MultiDimDeviceArray deviceArray = (MultiDimDeviceArray) args[argIdx];
                            UnsafeHelper.PointerObject pointer = UnsafeHelper.createPointerObject();
                            pointer.setValueOfPointer(deviceArray.getPointer());
                            kernelArgs.setArgument(argIdx, pointer);
                        } else {
                            CompilerDirectives.transferToInterpreter();
                            throw UnsupportedTypeException.create(new Object[]{args[argIdx]}, "expected DeviceArray type");
                        }
                        break;
                }
            } catch (UnsupportedMessageException e) {
                CompilerDirectives.transferToInterpreter();
                throw UnsupportedTypeException.create(new Object[]{args[argIdx]}, "expected type " + type);
            }
        }
        return kernelArgs;
    }

    private static ArgumentType[] parseSignature(String kernelSignature) {
        CompilerAsserts.neverPartOfCompilation();
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

    private static List<Boolean> computeIfArgsAreArrays(ArgumentType[] types) {
        return Arrays.stream(types).map(a -> a == ArgumentType.POINTER).collect(Collectors.toList());
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

    // implementation of InteropLibrary

    protected static final String PTX = "ptx";
    protected static final String NAME = "name";
    protected static final String LAUNCH_COUNT = "launchCount";
    private static final MemberSet MEMBERS = new MemberSet(PTX, NAME, LAUNCH_COUNT);

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

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String member) {
        return PTX.equals(member) || NAME.equals(member) || LAUNCH_COUNT.equals(member);
    }

    @ExportMessage
    @SuppressWarnings("unused")
    abstract static class ReadMember {
        @Specialization(guards = "PTX.equals(member)")
        public static String readMemberPtx(Kernel receiver, String member) {
            String ptx = receiver.getPTX();
            if (ptx == null) {
                return "<no PTX code>";
            } else {
                return ptx;
            }
        }

        @Specialization(guards = "NAME.equals(member)")
        public static String readMemberName(Kernel receiver, String member) {
            return receiver.getKernelName();
        }

        @Specialization(guards = "LAUNCH_COUNT.equals(member)")
        public static int readMemberLaunchCount(Kernel receiver, String member) {
            return receiver.getLaunchCount();
        }

        @Fallback
        public static Object readMemberOther(Kernel receiver, String member) throws UnknownIdentifierException {
            throw UnknownIdentifierException.create(member);
        }
    }

    private static int extractNumber(Object valueObj, String argumentName, InteropLibrary access) throws UnsupportedTypeException {
        try {
            return access.asInt(valueObj);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{valueObj}, "integer expected for " + argumentName);
        }
    }

    private static Dim3 extractDim3(Object valueObj, String argumentName, InteropLibrary access, InteropLibrary elementAccess) throws UnsupportedTypeException {
        if (access.hasArrayElements(valueObj)) {
            long size;
            try {
                size = access.getArraySize(valueObj);
            } catch (UnsupportedMessageException e) {
                CompilerDirectives.transferToInterpreter();
                throw new GrCUDAInternalException("unexpected behavior");
            }
            if (size < 1 || size > 3) {
                CompilerDirectives.transferToInterpreter();
                throw UnsupportedTypeException.create(new Object[]{valueObj}, argumentName + " needs to have between 1 and 3 elements");
            }
            int[] dim3 = new int[]{1, 1, 1};
            final char[] suffix = {'x', 'y', 'z'};
            for (int i = 0; i < size; i++) {
                Object elementObj;
                try {
                    elementObj = access.readArrayElement(valueObj, i);
                } catch (UnsupportedMessageException e) {
                    CompilerDirectives.transferToInterpreter();
                    throw new GrCUDAInternalException("unexpected behavior");
                } catch (InvalidArrayIndexException e) {
                    CompilerDirectives.transferToInterpreter();
                    throw UnsupportedTypeException.create(new Object[]{valueObj}, argumentName + " needs to have between 1 and 3 elements");
                }
                dim3[i] = extractNumber(elementObj, "dim3." + suffix[i], elementAccess);
            }
            return new Dim3(dim3[0], dim3[1], dim3[2]);
        }
        return new Dim3(extractNumber(valueObj, argumentName, access));
    }
    
    private static CUDAStream extractStream(Object streamObj) throws UnsupportedTypeException {
        if (streamObj instanceof CUDAStream) {
            return (CUDAStream) streamObj;
        } else {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{streamObj}, "expected CUDAStream type, received " + streamObj.getClass());
        }
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary(limit = "3") InteropLibrary gridSizeAccess,
                    @CachedLibrary(limit = "3") InteropLibrary gridSizeElementAccess,
                    @CachedLibrary(limit = "3") InteropLibrary blockSizeAccess,
                    @CachedLibrary(limit = "3") InteropLibrary blockSizeElementAccess,
                    @CachedLibrary(limit = "3") InteropLibrary sharedMemoryAccess) throws UnsupportedTypeException, ArityException {
        // FIXME: ArityException allows to specify only 1 arity, and cannot be sublassed! We might want to use a custom exception here;
        if (arguments.length < 2 || arguments.length > 4) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(2, arguments.length);
        }

        Dim3 gridSize = extractDim3(arguments[0], "gridSize", gridSizeAccess, gridSizeElementAccess);
        Dim3 blockSize = extractDim3(arguments[1], "blockSize", blockSizeAccess, blockSizeElementAccess);
        KernelConfigBuilder configBuilder = new KernelConfigBuilder(gridSize, blockSize);
        if (arguments.length == 3) {
            if (sharedMemoryAccess.isNumber(arguments[2])) {
                // Dynamic shared memory specified;
                configBuilder.dynamicSharedMemoryBytes(extractNumber(arguments[2], "dynamicSharedMemory", sharedMemoryAccess));
            } else {
                // Stream specified;
                configBuilder.stream(extractStream(arguments[2]));
            }
        } else if (arguments.length == 4) {
            configBuilder.dynamicSharedMemoryBytes(extractNumber(arguments[2], "dynamicSharedMemory", sharedMemoryAccess));
            // Stream specified;
            configBuilder.stream(extractStream(arguments[3]));
        }

        return new ConfiguredKernel(this, configBuilder.build());
    }
}


