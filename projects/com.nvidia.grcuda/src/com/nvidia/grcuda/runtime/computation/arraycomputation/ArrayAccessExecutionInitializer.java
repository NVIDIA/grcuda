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
package com.nvidia.grcuda.runtime.computation.arraycomputation;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.computation.InitializeDependencyList;

import java.util.Collections;
import java.util.List;

/**
 * The only argument in {@link com.nvidia.grcuda.runtime.array.AbstractArray} computations is the array itself.
 * Note that in {@link com.nvidia.grcuda.runtime.array.MultiDimDeviceArrayView} the array is the parent {@link com.nvidia.grcuda.runtime.array.MultiDimDeviceArray},
 * while in {@link com.nvidia.grcuda.runtime.array.MultiDimDeviceArray} there is currently no need to explicitly represent computations,
 * as they cannot directly access the underlying memory;
 */
class ArrayAccessExecutionInitializer<T extends AbstractArray> implements InitializeDependencyList {

    private final T array;
    private final boolean readOnly;
    private final static String PARAMETER_NAME = "array_access";

    ArrayAccessExecutionInitializer(T array, boolean readOnly) {
        this.array = array;
        this.readOnly = readOnly;
    }

    @Override
    public List<ComputationArgumentWithValue> initialize() {
        return Collections.singletonList(
                new ComputationArgumentWithValue(PARAMETER_NAME, Type.NFI_POINTER, this.readOnly ? ComputationArgument.Kind.POINTER_IN : ComputationArgument.Kind.POINTER_INOUT, this.array));
    }
}
