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

import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.GrCUDAException;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class BuildKernelFunction extends Function {
    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;

    public BuildKernelFunction(AbstractGrCUDAExecutionContext grCUDAExecutionContext) {
        super("buildkernel");
        this.grCUDAExecutionContext = grCUDAExecutionContext;
    }

    @Override
    @TruffleBoundary
    /**
     * Build kernel from source code using NVRTC and return callable.
     *
     * This call compiles the source immediately and not lazily after the first invocation. The
     * <code>buildkernel</code> call supports two different overloaded forms.
     *
     * 1. The legacy form with 3-arguments:
     * <code>buildkernel(sourceCode, kernelName, nfiSignatureString)<code>
     *    where <code>sourceCode</code> is the CUDA C/C++ source code string, `kernelName` the
     * non-lowered name of the kernel (possibly with template parameter), and
     * <code>nfiSignatureString</code>, which is like <code>uint64, pointer, pointer, float</code>.
     *
     *
     * 2. The NIDL version with 2-arguments:
     * <code>buildkernel(sourceCode, nidlSignatureString)</code> where
     * <code>nidlSignatureString</code> is like
     * <code>kernelName(param1: uint64, param2: in pointer float, param3: out pointer float, param4: float)</code>.
     * Note the kernel name can also contain a template instantiation argument if the kernel
     * function is templated, e.g.,
     * <code>increment<int>(arr: inout pointer sint32, length: sint32)</code>.
     *
     * @param arguments string of length 2 or 3 containing the arguments for function
     *            <code>buildkernel</code>.
     * @param Kernel object
     */
    public Object call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        if ((arguments.length != 2) && (arguments.length != 3)) {
            throw new GrCUDAException("bindkernel function requires two or three arguments");
        }

        String code = expectString(arguments[0], "argument 1 of buildkernel must be string (kernel code)");
        String parameterSignature = null;
        String kernelName = null;
        if (arguments.length == 3) {
            // parse 3 argument call with kernel name and legacy NFI-based kernel signature
            // (comma-separated NFI types)
            kernelName = expectString(arguments[1], "argument 2 of buildkernel must be string (kernel name)").trim();
            parameterSignature = expectString(arguments[2], "argument 3 of build must be string (signature of kernel)").trim();
        } else {
            // parse NIDL kernel signature
            //
            // kernelName<T>(argName_i: sint32, argName_j: inout pointer float)
            // -> get kernel "kernelName<T>"
            // -> effective kernel name "kernelName"
            //
            // kernelName(argName_i: sint32, argName_j: inout pointer float)
            // -> get kernel "kernelName"
            // -> effective kernel name "kernelName"
            //
            String signature = expectString(arguments[1], "argument 2 of bindkernel must be string (NIDL signature)").trim();
            KernelNameSignaturePair kernelNameSignaturePair = parseSignature(signature);
            kernelName = kernelNameSignaturePair.getKernelName();
            parameterSignature = kernelNameSignaturePair.getParameterSignature();
        }
        return grCUDAExecutionContext.buildKernel(code, kernelName, parameterSignature);
    }

    private static KernelNameSignaturePair parseSignature(String signature) {
        String s = signature.trim();
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
        String kernelName = s.substring(0, firstLParenPos).trim();
        String paramSignature = s.substring(firstLParenPos + 1, lastRParenPos).trim();
        return new KernelNameSignaturePair(kernelName, paramSignature);
    }

    private static final class KernelNameSignaturePair {
        private final String kernelName;
        private final String paramSignature;

        KernelNameSignaturePair(String kernelName, String paramSignature) {
            this.kernelName = kernelName;
            this.paramSignature = paramSignature;
        }

        String getKernelName() {
            return kernelName;
        }

        String getParameterSignature() {
            return paramSignature;
        }
    }
}
