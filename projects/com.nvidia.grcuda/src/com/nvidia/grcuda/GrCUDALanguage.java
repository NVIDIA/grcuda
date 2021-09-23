/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.nvidia.grcuda;


import com.oracle.truffle.api.TruffleLogger;
import org.graalvm.options.OptionDescriptors;

import com.nvidia.grcuda.nodes.ExpressionNode;
import com.nvidia.grcuda.nodes.GrCUDARootNode;
import com.nvidia.grcuda.parser.ParserAntlr;
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.Truffle;
import com.oracle.truffle.api.TruffleLanguage;


/**
 * GrCUDA Truffle language that exposes the GPU device and CUDA runtime to polyglot Graal languages.
 */
@TruffleLanguage.Registration(id = GrCUDALanguage.ID, name = "grcuda", version = "0.1", internal = false, contextPolicy = TruffleLanguage.ContextPolicy.SHARED)
public final class GrCUDALanguage extends TruffleLanguage<GrCUDAContext> {

    public static final String ID = "grcuda";

    public static final TruffleLogger LOGGER = TruffleLogger.getLogger(ID, "com.nvidia.grcuda");

    @Override
    protected GrCUDAContext createContext(Env env) {
        if (!env.isNativeAccessAllowed()) {
            throw new GrCUDAException("cannot create CUDA context without native access");
        }
        return new GrCUDAContext(env);
    }

    @Override
    protected CallTarget parse(ParsingRequest request) {
        ExpressionNode expression = new ParserAntlr().parse(request.getSource());
        GrCUDARootNode newParserRoot = new GrCUDARootNode(this, expression);
        return Truffle.getRuntime().createCallTarget(newParserRoot);
    }

    public static GrCUDALanguage getCurrentLanguage() {
        return TruffleLanguage.getCurrentLanguage(GrCUDALanguage.class);
    }

    public static GrCUDAContext getCurrentContext() {
        return getCurrentContext(GrCUDALanguage.class);
    }

    @Override
    protected void disposeContext(GrCUDAContext cxt) {
        cxt.disposeAll();
    }

    @Override
    protected OptionDescriptors getOptionDescriptors() {
        return new GrCUDAOptionsOptionDescriptors();
    }

    @Override
    protected boolean isThreadAccessAllowed(Thread thread, boolean singleThreaded) {
        return true;
    }

    @Override
    protected void finalizeContext(GrCUDAContext context) {
        context.cleanup();
    }
}
