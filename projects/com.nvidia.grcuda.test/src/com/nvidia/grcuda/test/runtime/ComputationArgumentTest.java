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
package com.nvidia.grcuda.test.runtime;

import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.TypeException;
import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;

public class ComputationArgumentTest {

    @Test
    public void testSignatureParsingOld() throws TypeException {
        String signature = "pointer, const pointer, double, sint32";
        Boolean[] isArray = {true, true, false, false};
        Type[] types = {Type.NFI_POINTER, Type.NFI_POINTER, Type.DOUBLE, Type.SINT32};
        ArrayList<ComputationArgument> params = ComputationArgument.parseParameterSignature(signature);
        for (int i = 0; i < params.size(); i++) {
            assertEquals(i, params.get(i).getPosition());
            assertEquals(isArray[i], params.get(i).isArray());
            assertEquals(types[i], params.get(i).getType());
        }
    }

    @Test
    public void testSignatureParsingOldWithConst() throws TypeException {
        String signature = "pointer, const pointer, double, sint32";
        Boolean[] isArray = {true, true, false, false};
        Boolean[] isConst = {false, true, true, true};
        Type[] types = {Type.NFI_POINTER, Type.NFI_POINTER, Type.DOUBLE, Type.SINT32};
        ArrayList<ComputationArgument> params = ComputationArgument.parseParameterSignature(signature);
        for (int i = 0; i < params.size(); i++) {
            assertEquals(i, params.get(i).getPosition());
            assertEquals(isArray[i], params.get(i).isArray());
            assertEquals(types[i], params.get(i).getType());
            assertEquals(isConst[i], params.get(i).isConst());
        }
    }

    @Test
    public void testSignatureParsingNIDL() throws TypeException {
        String signature = "x: in pointer sint32, y: inout pointer float, z: out pointer float, n: sint32, n_blocks: sint64, block_size: char";
        String[] names = {"x", "y", "z", "n", "n_blocks", "block_size"};
        Boolean[] isArray = {true, true, true, false, false, false};
        Boolean[] isConst = {true, false, false, true, true, true};
        Type[] types = {Type.SINT32, Type.FLOAT, Type.FLOAT, Type.SINT32, Type.SINT64, Type.CHAR};
        ArrayList<ComputationArgument> params = ComputationArgument.parseParameterSignature(signature);
        for (int i = 0; i < params.size(); i++) {
            assertEquals(names[i], params.get(i).getName());
            assertEquals(i, params.get(i).getPosition());
            assertEquals(isArray[i], params.get(i).isArray());
            assertEquals(types[i], params.get(i).getType());
            assertEquals(isConst[i], params.get(i).isConst());
        }
    }

    @Test
    public void testSignatureParsingWithParentheses() throws TypeException {
        String signature = "(sint64, sint32, pointer const, const pointer, sint32, pointer, sint32): sint32\"";
        Boolean[] isArray = {false, false, true, true, false, true, false};
        Boolean[] isConst = {true, true, true, true, true, false, true};
        Type[] types = {Type.SINT64, Type.SINT32, Type.NFI_POINTER, Type.NFI_POINTER, Type.SINT32, Type.NFI_POINTER, Type.SINT32};
        ArrayList<ComputationArgument> params = ComputationArgument.parseParameterSignature(signature);
        for (int i = 0; i < params.size(); i++) {
            assertEquals(i, params.get(i).getPosition());
            assertEquals(isArray[i], params.get(i).isArray());
            assertEquals(types[i], params.get(i).getType());
            assertEquals(isConst[i], params.get(i).isConst());
        }
    }

    @Test(expected = TypeException.class)
    public void testSignatureParsingWithWrongParentheses() throws TypeException {
        String signature = "(sint64, sint32, pointer const), const pointer, sint32, pointer, sint32): sint32\"";
        ComputationArgument.parseParameterSignature(signature);
    }
}
