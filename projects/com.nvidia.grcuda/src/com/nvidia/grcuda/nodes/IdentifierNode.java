/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.nvidia.grcuda.nodes;

import java.util.Optional;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDALanguage;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.functions.FunctionTable;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.CachedContext;
import com.oracle.truffle.api.dsl.Specialization;

public abstract class IdentifierNode extends ExpressionNode {

    private final String identifierName;
    private final String namespace;

    public IdentifierNode(String identifierName) {
        this.identifierName = identifierName;
        this.namespace = "";
    }

    public IdentifierNode(String identifierName, String namespace) {
        this.identifierName = identifierName;
        this.namespace = namespace;
    }

    public String getIdentifierName() {
        return identifierName;
    }

    public String getNamespace() {
        return namespace;
    }

    @Specialization
    protected Object doDefault(
                    @CachedContext(GrCUDALanguage.class) GrCUDAContext context) {
        FunctionTable functionTable = context.getFunctionTable();
        Optional<Function> maybeFunction = functionTable.lookupFunction(identifierName, namespace);
        if (!maybeFunction.isPresent()) {
            CompilerDirectives.transferToInterpreter();
            throw new RuntimeException("Function '" + identifierName + "' not found in namespace '" + namespace + "'");
        }
        return maybeFunction.get();
    }
}
