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
package com.nvidia.grcuda.runtime.computation;

import com.nvidia.grcuda.GrCUDALogger;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.LibrarySetStream;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static com.nvidia.grcuda.functions.Function.INTEROP;

/**
 * Computational element that wraps calls to CUDA libraries such as cuBLAS or cuML.
 */
public class CUDALibraryExecution extends GrCUDAComputationalElement {

    private final Function nfiFunction;
    protected Object[] argsWithHandle;
    private final LibrarySetStream setStreamFunctionNFI;

    public CUDALibraryExecution(AbstractGrCUDAExecutionContext context, Function nfiFunction, LibrarySetStream setStreamFunctionNFI, List<ComputationArgumentWithValue> args) {
        this(context, nfiFunction, setStreamFunctionNFI, args, 0);
    }

    public CUDALibraryExecution(AbstractGrCUDAExecutionContext context, Function nfiFunction, LibrarySetStream setStreamFunctionNFI, List<ComputationArgumentWithValue> args, int extraArguments) {
        super(context, new CUDALibraryExecutionInitializer(args));
        this.nfiFunction = nfiFunction;
        this.setStreamFunctionNFI = setStreamFunctionNFI;

        // Array of [libraryHandle + arguments], required by CUDA libraries for execution.
        // Some libraries (such as cuSPARSE) wrap input arrays, making it not possible to directly track them.
        // We add those arrays at the end of "args" so they are tracked by CUDALibraryExecutionInitializer,
        // but remove them here so the list of arguments passed to the final CUDA library function is correct;
        this.argsWithHandle = new Object[args.size() - extraArguments];
        for (int i = 0; i < args.size() - extraArguments; i++) {
            argsWithHandle[i] = args.get(i).getArgumentValue();
        }
    }

    @Override
    public boolean canUseStream() {
        return true;
    }

    @Override
    public Object execute() throws UnsupportedTypeException {
        // Execution happens on the default stream;
        Object result = null;
        try {
            this.setStreamFunctionNFI.setStream(this.getStream());
            result = INTEROP.execute(this.nfiFunction, this.argsWithHandle);
        } catch (ArityException | UnsupportedMessageException e) {
            GrCUDALogger.getLogger(GrCUDALogger.COMPUTATION_LOGGER).severe("error in execution of the function");
            e.printStackTrace();
        }
        return result;
    }

    static class CUDALibraryExecutionInitializer implements InitializeDependencyList {
        private final List<ComputationArgumentWithValue> args;

        CUDALibraryExecutionInitializer(List<ComputationArgumentWithValue> args) {
            this.args = args;
        }

        @Override
        public List<ComputationArgumentWithValue> initialize() {
            // Consider only arrays as dependencies;
            // The CUDA documentation is not clear on whether you can have concurrent computations
            // with the same handle;
            return this.args.stream().filter(ComputationArgument::isArray).collect(Collectors.toList());
        }
    }
}
