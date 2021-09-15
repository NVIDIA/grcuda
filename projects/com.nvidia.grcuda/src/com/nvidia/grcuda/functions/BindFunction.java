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

import java.util.ArrayList;

import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.FunctionBinding;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDALanguage;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.TypeException;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public final class BindFunction extends Function {

    public BindFunction() {
        super("bind");
    }

    @Override
    @TruffleBoundary
    /**
     * Bind to host function symbol from shared library.
     *
     * This call resolves the symbol immediately and not lazily after the first invocation. The
     * <code>bind</code> call supports two different overloaded forms.
     *
     * 1. The legacy form with 3-arguments:
     * <code>bind(libFileName, symbolName, nfiSignatureString)<code>
     *    where <code>symbolName</code> is the name of the symbol as it appears and the shared
     * library (possibly mangled) and <code>nfiSignatureString</code> is like
     * <code>(uint64, pointer, pointer, float): sint32</code>.
     *
     *
     * 2. The NIDL version with 2-arguments: <code>bind(libFileName, nidlSignatureString)</code>
     * where <code>nidlSignatureString</code> is like
     * <code>functionName(param1: uint64, param2: in pointer float, param3: out pointer float, param4: float): sint32</code>.
     * If the host function symbol is C++ mangled, prefix the keyword <code>cxx</code>, i.e.,
     * <code>cxx functionName(...)</code>.
     *
     * @param arguments string of length 2 or 3 containing the arguments for function
     *            <code>bind</code>.
     * @param HostFunction object
     */
    public HostFunction call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        if ((arguments.length != 2) && (arguments.length != 3)) {
            throw new GrCUDAException("bind function requires two or three arguments");
        }
        String libraryFile = expectString(arguments[0], "argument 1 of bind must be string (library filename)");
        String signature = (arguments.length == 2) ? expectString(arguments[1], "argument 2 of bind must be string (NIDL signature)").trim()
                        : expectString(arguments[2], "argument 3 of bind must be string (signature)").trim();
        String symbolName = (arguments.length == 3) ? expectString(arguments[1], "argument 2 of bind must be string (symbol name)").trim() : "";

        FunctionBinding binding = parseSignature(symbolName + signature);
        binding.setLibraryFileName(libraryFile);
        HostFunction hf = new HostFunction(binding, GrCUDALanguage.getCurrentContext().getCUDARuntime());
        try {
            hf.resolveSymbol();
        } catch (UnknownIdentifierException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(binding.getSymbolName() + " not found in " + libraryFile);
        }
        return hf;
    }

    private static FunctionBinding parseSignature(String signature) {
        String s = signature.trim();
        boolean isCxxSymbol = false;
        if (s.startsWith("cxx ")) {
            s = s.substring("cxx ".length());
            isCxxSymbol = true;
        }
        int firstLParenPos = s.indexOf('(');
        if (firstLParenPos == -1) {
            throw new GrCUDAException("expected \"(\"");
        }
        if (firstLParenPos == 0) {
            throw new GrCUDAException("expected identifier name before \"(\"");
        }
        int lastRParenPos = s.lastIndexOf(')');
        if (lastRParenPos == -1) {
            throw new GrCUDAException("expected \")\"");
        }
        if (lastRParenPos <= firstLParenPos) {
            throw new GrCUDAException("expected valid parameter signature within \"(  )\"");
        }
        String name = s.substring(0, firstLParenPos).trim();
        String parenSignature = s.substring(firstLParenPos + 1, lastRParenPos).trim();
        int typeColonPos = s.indexOf(':', lastRParenPos);
        if (typeColonPos == -1 || (s.length() - 1 - typeColonPos) <= 1) {
            throw new GrCUDAException("expected \":\" and return type");
        }
        String returnTypeString = s.substring(typeColonPos + 1).trim();
        try {
            Type returnType = Type.fromNIDLTypeString(returnTypeString);
            ArrayList<ComputationArgument> paramList = ComputationArgument.parseParameterSignature(parenSignature);
            if (isCxxSymbol) {
                return FunctionBinding.newCxxBinding(name, paramList, returnType);
            } else {
                return FunctionBinding.newCBinding(name, paramList, returnType);
            }
        } catch (TypeException e) {
            throw new GrCUDAException("invalid type: " + e.getMessage());
        }
    }
}
