/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
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

import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.MemberSet;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.runtime.CUDARuntime.CUModule;
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

@ExportLibrary(InteropLibrary.class)
public class Kernel implements TruffleObject {

    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final String kernelName;
    private final String kernelSymbol;
    private final List<Long> nativeKernelFunctionHandle;
    private final List<CUModule> modules;
    private final ComputationArgument[] kernelComputationArguments;
    private int launchCount = 0;
    private final String ptxCode;

    /**
     * Create a kernel without PTX code.
     *
     * @param grCUDAExecutionContext captured reference to the GrCUDA execution context
     * @param kernelName name of the kernel as exposed through Truffle
     * @param kernelSymbol name of the kernel symbol*
     * @param kernelFunction native pointer to the kernel function (CUfunction), one pointer for
     *            each device on which it is loaded
     * @param kernelSignature signature string of the kernel (NFI or NIDL)
     * @param modules CUmodules that contains the kernel function, one for each device
     */
    public Kernel(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String kernelName,
                    String kernelSymbol, List<Long> kernelFunction,
                    String kernelSignature, List<CUModule> modules) {
        this(grCUDAExecutionContext, kernelName, kernelSymbol, kernelFunction, kernelSignature, modules, "");
    }

    /**
     * Create a kernel and hold on to the PTX code.
     *
     * @param grCUDAExecutionContext captured reference to the GrCUDA execution context
     * @param kernelName name of kernel as exposed through Truffle
     * @param kernelSymbol name of the kernel symbol
     * @param kernelFunction native pointer to the kernel function (CUfunction), one pointer for
     *            each device on which it is loaded
     * @param kernelSignature signature string of the kernel (NFI or NIDL)
     * @param modules CUmodules that contains the kernel function, one for each device
     * @param ptx PTX source code for the kernel.
     */
    public Kernel(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String kernelName, String kernelSymbol,
                    List<Long> kernelFunction, String kernelSignature, List<CUModule> modules, String ptx) {
        try {
            List<ComputationArgument> paramList = ComputationArgument.parseParameterSignature(kernelSignature);
            ComputationArgument[] params = new ComputationArgument[paramList.size()];
            this.kernelComputationArguments = paramList.toArray(params);
        } catch (TypeException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(e.getMessage());
        }
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.kernelName = kernelName;
        this.kernelSymbol = kernelSymbol;
        this.nativeKernelFunctionHandle = kernelFunction;
        this.modules = modules;
        this.ptxCode = ptx;
        this.grCUDAExecutionContext.registerKernel(this);
    }

    public void incrementLaunchCount() {
        launchCount++;
    }

    public AbstractGrCUDAExecutionContext getGrCUDAExecutionContext() {
        return grCUDAExecutionContext;
    }

    public ComputationArgument[] getKernelParameters() {
        return kernelComputationArguments;
    }

    public long getKernelFunctionHandle(int deviceId) {
        if (modules.get(deviceId).isClosed()) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("CUmodule containing kernel " + kernelName + " is already closed");
        }
        return nativeKernelFunctionHandle.get(deviceId);
    }

    @Override
    public String toString() {
        return "Kernel(" + kernelName + ", " + Arrays.toString(kernelComputationArguments) + ", launchCount=" + launchCount + ")";
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
        if (arguments.length < 2 || arguments.length > 4) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(2, 4, arguments.length);
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
