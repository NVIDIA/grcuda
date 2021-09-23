/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

package com.nvidia.grcuda.test;

import org.junit.Test;
import com.nvidia.grcuda.parser.ParserAntlr;
import com.nvidia.grcuda.parser.GrCUDAParserException;
import com.oracle.truffle.api.source.Source;

public class ParserTest {

    @Test
    public void testArrayExpressionWithIntLiteral() throws GrCUDAParserException {
        parseString("double[1000]");
    }

    @Test
    public void testArrayExpressionWithIntExpr() throws GrCUDAParserException {
        parseString("double[(100+200)*4]");
    }

    @Test(expected = GrCUDAParserException.class)
    public void testArrayExpressionWithEmptyBracketsFails() throws GrCUDAParserException {
        parseString("double[]");
    }

    @Test(expected = GrCUDAParserException.class)
    public void testArrayExpressionWithIdentifierInBracketsFails() throws GrCUDAParserException {
        parseString("double[foo]");
    }

    @Test
    public void testNoArgsCall() throws GrCUDAParserException {
        parseString("foo()");
        parseString("foo( )");
    }

    @Test
    public void testBindCall() throws GrCUDAParserException {
        parseString("bind(\"libml.so\", \"dbscanFitDouble\", \"(pointer, sint32, sint32, double, sint32, pointer): void\")");
    }

    @Test
    public void testBuiltinCallable() throws GrCUDAParserException {
        parseString("cudaMallocManaged");
    }

    @Test
    public void testBuiltinCallableColonColon() throws GrCUDAParserException {
        parseString("::cudaMallocManaged");
    }

    @Test
    public void testCallableInNamespace() throws GrCUDAParserException {
        parseString("ML::dbscanFitDouble");
    }

    @SuppressWarnings("static-method")
    private void parseString(String sourceStr) throws GrCUDAParserException {
        Source source = Source.newBuilder("cuda", sourceStr, "testsource").build();
        new ParserAntlr().parse(source);
    }
}
