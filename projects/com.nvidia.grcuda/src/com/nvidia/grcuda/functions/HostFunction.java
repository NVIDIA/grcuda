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
package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.FunctionBinding;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class HostFunction extends Function {

    private final CUDARuntime cudaRuntime;
    private final FunctionBinding binding;
    private Object nfiCallable = null;

    public HostFunction(FunctionBinding binding, CUDARuntime runtime) {
        super(binding.getName());
        this.binding = binding;
        this.cudaRuntime = runtime;
    }

    @Override
    @TruffleBoundary
    protected Object call(Object[] arguments) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        assertFunctionResolved();
        return INTEROP.execute(nfiCallable, arguments);
    }

    @Override
    public String toString() {
        return "HostFunction(name=" + binding.getName() + ", nfiCallable=" + nfiCallable + ")";
    }

    public void resolveSymbol() throws UnknownIdentifierException {
        synchronized (this) {
            if (nfiCallable == null) {
                nfiCallable = cudaRuntime.getSymbol(binding);
                assert nfiCallable != null : "NFI callable non-null";
            }
        }
    }

    private void assertFunctionResolved() {
        synchronized (this) {
            if (nfiCallable == null) {
                try {
                    nfiCallable = cudaRuntime.getSymbol(binding);
                } catch (UnknownIdentifierException e) {
                    throw new GrCUDAException("symbol " + binding.getSymbolName() + " not found: " + e);
                }
                assert nfiCallable != null : "NFI callable non-null";
            }
        }
    }
}
