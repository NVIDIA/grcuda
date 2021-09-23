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
package com.nvidia.grcuda.nodes;

import java.util.ArrayList;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.GrCUDALanguage;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.MultiDimDeviceArray;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.CachedContext;
import com.oracle.truffle.api.dsl.Specialization;
import com.oracle.truffle.api.frame.VirtualFrame;

public abstract class ArrayNode extends ExpressionNode {

    @Children private ExpressionNode[] sizeNodes;

    private final Type elementType;

    ArrayNode(Type elementType, ArrayList<ExpressionNode> sizeNodes) {
        this.elementType = elementType;
        this.sizeNodes = new ExpressionNode[sizeNodes.size()];
        sizeNodes.toArray(this.sizeNodes);
    }

    @Specialization
    AbstractArray doDefault(VirtualFrame frame,
                            @CachedContext(GrCUDALanguage.class) GrCUDAContext context) {
        final AbstractGrCUDAExecutionContext grCUDAExecutionContext = context.getGrCUDAExecutionContext();
        long[] elementsPerDim = new long[sizeNodes.length];
        int dim = 0;
        for (ExpressionNode sizeNode : sizeNodes) {
            Object size = sizeNode.execute(frame);
            if (!(size instanceof Number)) {
                CompilerDirectives.transferToInterpreter();
                throw new GrCUDAInternalException("size in dimension " + dim + " must be a number", this);
            }
            elementsPerDim[dim] = ((Number) size).longValue();
            dim += 1;
        }
        if (sizeNodes.length == 1) {
            return new DeviceArray(grCUDAExecutionContext, elementsPerDim[0], elementType);
        } else {
            final boolean columnMajorOrder = false;
            return new MultiDimDeviceArray(grCUDAExecutionContext, elementType, elementsPerDim, columnMajorOrder);
        }
    }
}
