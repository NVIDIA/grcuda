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
package com.nvidia.grcuda.cudalibraries;

import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.functions.Function;
import com.oracle.truffle.api.CompilerDirectives;

import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper class to CUDA library functions. It holds the signature of the function being wrapped,
 * and creates {@link ComputationArgument} for the signature and inputs;
 */
public abstract class CUDALibraryFunction extends Function {

    protected final List<ComputationArgument> computationArguments;

    /**
     * Constructor, it takes the name of the wrapped function and its NFI signature,
     * and creates a list of {@link ComputationArgument} from it;
     * @param name name of the function
     * @param nfiSignature NFI signature of the function
     */
    protected CUDALibraryFunction(String name, String nfiSignature) {
        super(name);
        // Create the list of computation arguments;
        try {
            this.computationArguments = ComputationArgument.parseParameterSignature(nfiSignature);
        } catch (TypeException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(e.getMessage());
        }
    }

    /**
     * Given a list of inputs, map each signature argument to the corresponding input.
     * Assume that inputs are given in the same order as specified by the signature.
     * In this case, provide a pointer to the CUDA library handle, which is stored as first parameter of the argument list
     * @param args list of inputs
     * @param libraryHandle pointer to the native object used as CUDA library handle
     * @return list of inputs mapped to signature elements, used to compute dependencies
     */
    public List<ComputationArgumentWithValue> createComputationArgumentWithValueList(Object[] args, Long libraryHandle) {
        ArrayList<ComputationArgumentWithValue> argumentsWithValue = new ArrayList<>();
        // Set the library handle;
        argumentsWithValue.add(new ComputationArgumentWithValue(this.computationArguments.get(0), libraryHandle));
        // Set the other arguments;
        for (int i = 0; i < args.length; i++) {
            argumentsWithValue.add(new ComputationArgumentWithValue(this.computationArguments.get(i + 1), args[i]));
        }
        return argumentsWithValue;
    }
}

