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

import com.nvidia.grcuda.runtime.computation.ComputationArgument;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Collectors;

public final class FunctionBinding extends Binding {

    private final Type returnType;

    private FunctionBinding(String name, ArrayList<ComputationArgument> computationArgumentList,
                    Type returnType, boolean hasCxxMangledName) {
        super(name, computationArgumentList, hasCxxMangledName);
        this.returnType = returnType;
    }

    public static FunctionBinding newCxxBinding(String name, ArrayList<ComputationArgument> computationArgumentList, Type returnType) {
        return new FunctionBinding(name, computationArgumentList, returnType, true);
    }

    public static FunctionBinding newCBinding(String name, ArrayList<ComputationArgument> computationArgumentList, Type returnType) {
        return new FunctionBinding(name, computationArgumentList, returnType, false);
    }

    @Override
    public String toString() {
        return super.toString() + " : " + returnType.toString().toLowerCase();
    }

    @Override
    public String toNIDLString() {
        return name + "(" + getNIDLParameterSignature() + "): " + returnType.toString().toLowerCase();
    }

    public String toNFISignature() {
        return "(" + Arrays.stream(computationArguments).map(ComputationArgument::toNFISignatureElement).collect(Collectors.joining(", ")) + "): " + returnType.getNFITypeName();
    }
}
