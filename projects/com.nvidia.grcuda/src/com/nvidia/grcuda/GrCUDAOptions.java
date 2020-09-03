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
package com.nvidia.grcuda;

import org.graalvm.options.OptionCategory;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionStability;

import com.nvidia.grcuda.cublas.CUBLASRegistry;
import com.nvidia.grcuda.cuml.CUMLRegistry;
import com.oracle.truffle.api.Option;

@Option.Group(GrCUDALanguage.ID)
public final class GrCUDAOptions {
    private GrCUDAOptions() {
        // no instances
    }

    @Option(category = OptionCategory.USER, help = "Enable cuBLAS support.", stability = OptionStability.STABLE) //
    public static final OptionKey<Boolean> CuBLASEnabled = new OptionKey<>(true);

    @Option(category = OptionCategory.USER, help = "Set the location of the cublas library.", stability = OptionStability.STABLE) //
    public static final OptionKey<String> CuBLASLibrary = new OptionKey<>(CUBLASRegistry.DEFAULT_LIBRARY);

    @Option(category = OptionCategory.USER, help = "Enable cuML support.", stability = OptionStability.STABLE) //
    public static final OptionKey<Boolean> CuMLEnabled = new OptionKey<>(true);

    @Option(category = OptionCategory.USER, help = "Set the location of the cuml library.", stability = OptionStability.STABLE) //
    public static final OptionKey<String> CuMLLibrary = new OptionKey<>(CUMLRegistry.DEFAULT_LIBRARY);

    @Option(category = OptionCategory.USER, help = "Choose the scheduling policy of GrCUDA computations", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> ExecutionPolicy = new OptionKey<>(GrCUDAContext.DEFAULT_EXECUTION_POLICY.getName());

    @Option(category = OptionCategory.USER, help = "Choose how data dependencies between GrCUDA computations are computed", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> DependencyPolicy = new OptionKey<>(GrCUDAContext.DEFAULT_DEPENDENCY_POLICY.getName());

    @Option(category = OptionCategory.USER, help = "Choose how streams for new GrCUDA computations are created", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> RetrieveNewStreamPolicy = new OptionKey<>(GrCUDAContext.DEFAULT_RETRIEVE_STREAM_POLICY.getName());

    @Option(category = OptionCategory.USER, help = "Choose how streams for new GrCUDA computations are obtained from parent computations", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> RetrieveParentStreamPolicy = new OptionKey<>(GrCUDAContext.DEFAULT_PARENT_STREAM_POLICY.getName());

    @Option(category = OptionCategory.USER, help = "Force the use of array stream attaching even when not required (e.g. post-Pascal GPUs)", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<Boolean> ForceStreamAttach = new OptionKey<>(false);
}
