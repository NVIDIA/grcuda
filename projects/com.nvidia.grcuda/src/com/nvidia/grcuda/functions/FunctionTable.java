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

import java.util.HashMap;
import java.util.Optional;

import com.nvidia.grcuda.GrCUDAInternalException;
import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;

public final class FunctionTable {

    private final HashMap<String, Function> functionMap = new HashMap<>();

    public FunctionTable registerFunction(Function function) {
        String functionName = function.getName();
        String namespace = function.getNamespace();
        String key = getKeyFromName(functionName, namespace);
        if (functionMap.containsKey(key)) {
            throw new GrCUDAInternalException("function '" + namespace + "::" + functionName + "' already exists.");
        }
        functionMap.put(key, function);
        return this;
    }

    @TruffleBoundary
    public Optional<Function> lookupFunction(String functionName, String namespace) {
        // A TruffleBoundary stops forced-inlining.
        // Here it is required because of the access to the HashMap, which uses
        // recursive method calls in the lookup. If the inlining is not stopped, the Graal
        // would continue infinitely.
        return Optional.ofNullable(functionMap.get(getKeyFromName(functionName, namespace)));
    }

    private static String getKeyFromName(String functionName, String namespace) {
        CompilerAsserts.neverPartOfCompilation();
        return namespace + "::" + functionName;
    }
}
