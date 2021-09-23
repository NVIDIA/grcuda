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
package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import java.util.ArrayList;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.KernelBinding;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.runtime.Kernel;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public final class BindKernelFunction extends Function {

    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;

    public BindKernelFunction(AbstractGrCUDAExecutionContext grCUDAExecutionContext) {
        super("bindkernel");
        this.grCUDAExecutionContext = grCUDAExecutionContext;
    }

    @Override
    @TruffleBoundary
    /**
     * Bind to kernel function symbol from cubin or PTX file.
     *
     * This call resolves the symbol immediately and not lazily after the first invocation. The
     * <code>bindkernel</code> call supports two different overloaded forms.
     *
     * 1. The legacy form with 3-arguments:
     * <code>bindkernel(fileName, symbolName, nfiSignatureString)<code>
     *    where <code>symbolName</code> is the name of the symbol as it appears and the cubin or PTX
     * file (possibly mangled) and <code>nfiSignatureString</code> is like
     * <code>uint64, pointer, pointer, float</code>.
     *
     *
     * 2. The NIDL version with 2-arguments: <code>bindkernel(fileName, nidlSignatureString)</code>
     * where <code>nidlSignatureString</code> is like
     * <code>kernelName(param1: uint64, param2: in pointer float, param3: out pointer float, param4: float)</code>.
     * If the kernel symbol is C++ mangled, prefix the keyword <code>cxx</code>, i.e.,
     * <code>cxx kernelName(...)</code>.
     *
     * @param arguments string of length 2 or 3 containing the arguments for function
     *            <code>bindkernel</code>.
     * @param Kernel object
     */
    public Kernel call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        if ((arguments.length != 2) && (arguments.length != 3)) {
            throw new GrCUDAException("bindkernel function requires two or three arguments");
        }
        String fileName = expectString(arguments[0], "argument 1 of bindkernel must be string (name of cubin or PTX file)");
        KernelBinding binding = null;
        if (arguments.length == 3) {
            // parse legacy NFI-based kernel signature: comma-separated NFI types
            String symbolName = expectString(arguments[1], "argument 2 of bindkernel must be string (symbol name)").trim();
            String signature = expectString(arguments[2], "argument 3 of bind must be string (signature of kernel)").trim();
            try {
                ArrayList<ComputationArgument> paramList = ComputationArgument.parseParameterSignature(signature);
                binding = KernelBinding.newCBinding(symbolName, paramList);
            } catch (TypeException e) {
                throw new GrCUDAException("invalid type: " + e.getMessage());
            }
        } else {
            // parse NIDL kernel signature
            //
            // kernelName(argName_i: sint32, argName_j: inout pointer float)
            // -> search for symbol "kernelName"
            //
            // cxx kernelName(argName_i: sint32, argName_j: inout pointer float)
            // -> search for symbol "_Z10kernelNameiPf"
            String signature = expectString(arguments[1], "argument 2 of bindkernel must be string (NIDL signature)").trim();
            binding = parseSignature(signature);
        }
        binding.setLibraryFileName(fileName);
        return grCUDAExecutionContext.loadKernel(binding);
    }

    private static KernelBinding parseSignature(String signature) {
        String s = signature.trim();
        boolean isCxxSymbol = false;
        if (s.startsWith("cxx ")) {
            s = s.substring("cxx ".length());
            isCxxSymbol = true;
        }
        int firstLParenPos = s.indexOf('(');
        if (firstLParenPos == -1) {
            // attempt to parse a
            throw new GrCUDAException("expected \"(\"");
        }
        if (firstLParenPos == 0) {
            throw new GrCUDAException("expected identifier name before \"(\"");
        }
        int lastRParenPos = s.lastIndexOf(')');
        if (lastRParenPos == -1) {
            throw new GrCUDAException("expected \")\"");
        }
        if (lastRParenPos != (s.length() - 1)) {
            throw new GrCUDAException("expected valid parameter signature within \"(  )\"");
        }
        String name = s.substring(0, firstLParenPos).trim();
        String parenSignature = s.substring(firstLParenPos + 1, lastRParenPos).trim();

        try {
            ArrayList<ComputationArgument> paramList = ComputationArgument.parseParameterSignature(parenSignature);
            if (isCxxSymbol) {
                return KernelBinding.newCxxBinding(name, paramList);
            } else {
                return KernelBinding.newCBinding(name, paramList);
            }
        } catch (TypeException e) {
            throw new GrCUDAException("invalid type: " + e.getMessage());
        }
    }
}
