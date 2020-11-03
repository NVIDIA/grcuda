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
import com.nvidia.grcuda.MemberSet;
import com.nvidia.grcuda.array.MultiDimDeviceArray;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.CUDAStream;

import java.util.ArrayList;
import java.util.Arrays;

import com.nvidia.grcuda.Parameter;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.gpu.CUDARuntime.CUModule;
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

import java.util.List;

@ExportLibrary(InteropLibrary.class)
public class Kernel implements TruffleObject {

    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final String kernelName;
    private final String kernelSymbol;
    private final long nativeKernelFunctionHandle;
    private final CUModule module;
    private final Parameter[] kernelParameters;
    private int launchCount = 0;
    private String ptxCode;

    /**
     * Create a kernel without PTX code.
     *
     * @param grCUDAExecutionContext captured reference to the GrCUDA execution context
     * @param kernelName name of the kernel as exposed through Truffle
     * @param kernelSymbol name of the kernel symbol*
     * @param kernelFunction native pointer to the kernel function (CUfunction)
     * @param kernelSignature signature string of the kernel (NFI or NIDL)
     * @param module CUmodule that contains the kernel function
     */
    public Kernel(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String kernelName,
                    String kernelSymbol, long kernelFunction,
                    String kernelSignature, CUModule module) {
        this(grCUDAExecutionContext, kernelName, kernelSymbol, kernelFunction, kernelSignature, module, "");
    }

    /**
     * Create a kernel and hold on to the PTX code.
     *
     * @param grCUDAExecutionContext captured reference to the GrCUDA execution context
     * @param kernelName name of kernel as exposed through Truffle
     * @param kernelSymbol name of the kernel symbol
     * @param kernelFunction native pointer to the kernel function (CUfunction)
     * @param kernelSignature signature string of the kernel (NFI or NIDL)
     * @param module CUmodule that contains the kernel function
     * @param ptx PTX source code for the kernel.
     */
    public Kernel(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String kernelName, String kernelSymbol,
                    long kernelFunction, String kernelSignature, CUModule module, String ptx) {
//        parseSignature(kernelSignature);
        try {
            ArrayList<Parameter> paramList = Parameter.parseParameterSignature(kernelSignature);
            Parameter[] params = new Parameter[paramList.size()];
            this.kernelParameters = paramList.toArray(params);
        } catch (TypeException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(e.getMessage());
        }
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.kernelName = kernelName;
        this.kernelSymbol = kernelSymbol;
        this.nativeKernelFunctionHandle = kernelFunction;
        this.module = module;
        this.ptxCode = ptx;
        this.grCUDAExecutionContext.registerKernel(this);
    }

    public void incrementLaunchCount() {
        launchCount++;
    }

    public AbstractGrCUDAExecutionContext getGrCUDAExecutionContext() {
        return grCUDAExecutionContext;
    }

    public Parameter[] getKernelParameters() {
        return kernelParameters;
    }

    KernelArguments createKernelArguments(Object[] args, InteropLibrary booleanAccess,
                    InteropLibrary int8Access, InteropLibrary int16Access,
                    InteropLibrary int32Access, InteropLibrary int64Access, InteropLibrary doubleAccess)
                    throws UnsupportedTypeException, ArityException {
        if (args.length != kernelParameters.length) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(kernelParameters.length, args.length);
        }
        KernelArguments kernelArgs = new KernelArguments(args, this.kernelParameters);
        for (int paramIdx = 0; paramIdx < kernelParameters.length; paramIdx++) {
            Object arg = args[paramIdx];
            Parameter param = kernelParameters[paramIdx];
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

//    private void parseSignature(String kernelSignature) {
//        for (String s : kernelSignature.trim().split(",")) {
//
//            // Find if the type is const;
//            String[] typePieces = s.trim().split(" ");
//            String typeString;
//            boolean typeIsConst = false;
//            if (typePieces.length == 1) {
//                // If only 1 piece is found, the argument is not const;
//                typeString = typePieces[0].trim();
//            } else if (typePieces.length == 2) {
//                // Const can be either before or after the type;
//                if (typePieces[0].trim().equals("const")) {
//                    typeIsConst = true;
//                    typeString = typePieces[1].trim();
//                } else if (typePieces[1].trim().equals("const")) {
//                    typeIsConst = true;
//                    typeString = typePieces[0].trim();
//                } else {
//                    throw new IllegalArgumentException("invalid type identifier in kernel signature: " + s);
//                }
//            } else {
//                throw new IllegalArgumentException("invalid type identifier in kernel signature: " + s);
//            }
//
//            ArgumentType type;
//            switch (typeString) {
//                case "pointer":
//                    type = ArgumentType.POINTER;
//                    break;
//                case "uint64":
//                case "sint64":
//                    type = ArgumentType.INT64;
//                    break;
//                case "uint32":
//                case "sint32":
//                    type = ArgumentType.INT32;
//                    break;
//                case "float":
//                    type = ArgumentType.FLOAT32;
//                    break;
//                case "double":
//                    type = ArgumentType.FLOAT64;
//                    break;
//                default:
//                    throw new IllegalArgumentException("invalid type identifier in kernel signature: " + s);
//            }
//            this.arguments.add(new ComputationArgument(type, type.equals(ArgumentType.POINTER), typeIsConst));
//        }
//    }

    private static GrCUDAException createExceptionValueOutOfRange(Type type, long value) {
        return new GrCUDAException("value " + value + " is out of range for type " + type);
    }

    public long getKernelFunctionHandle() {
        if (module.isClosed()) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("CUmodule containing kernel " + kernelName + " is already closed");
        }
        return nativeKernelFunctionHandle;
    }

    @Override
    public String toString() {
        return "Kernel(" + kernelName + ", " + Arrays.toString(kernelParameters) + ", launchCount=" + launchCount + ")";
    }

    public String getPTX() {
        return ptxCode;
    }

    public String getKernelName() {
        return kernelName;
    }

    public String getSymbolName() {
        return kernelSymbol;
    }

    public int getLaunchCount() {
        return launchCount;
    }

    // implementation of InteropLibrary

    protected static final String PTX = "ptx";
    protected static final String NAME = "name";
    protected static final String LAUNCH_COUNT = "launchCount";
    static final MemberSet MEMBERS = new MemberSet(PTX, NAME, LAUNCH_COUNT);

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
        // FIXME: ArityException allows to specify only 1 arity, and cannot be subclassed! We might want to use a custom exception here;
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


