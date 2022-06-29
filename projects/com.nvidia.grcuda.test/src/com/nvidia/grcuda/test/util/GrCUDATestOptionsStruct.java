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
package com.nvidia.grcuda.test.util;

import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;

public class GrCUDATestOptionsStruct {
    public final ExecutionPolicyEnum policy;
    public final boolean inputPrefetch;
    public final RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy;
    public final RetrieveParentStreamPolicyEnum retrieveParentStreamPolicy;
    public final DependencyPolicyEnum dependencyPolicy;
    public final DeviceSelectionPolicyEnum deviceSelectionPolicy;
    public final boolean forceStreamAttach;
    public final boolean timeComputation;
    public final int numberOfGPUs;

    /**
     * A simple struct that holds a combination of GrCUDA options, extracted from the output of {@link GrCUDATestUtil#getAllOptionCombinationsSingleGPU}
     */
    public GrCUDATestOptionsStruct(ExecutionPolicyEnum policy,
                                   boolean inputPrefetch,
                                   RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy,
                                   RetrieveParentStreamPolicyEnum retrieveParentStreamPolicy,
                                   DependencyPolicyEnum dependencyPolicy,
                                   DeviceSelectionPolicyEnum deviceSelectionPolicy,
                                   boolean forceStreamAttach,
                                   boolean timeComputation,
                                   int numberOfGPUs) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
        this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
        this.retrieveParentStreamPolicy = retrieveParentStreamPolicy;
        this.dependencyPolicy = dependencyPolicy;
        this.deviceSelectionPolicy = deviceSelectionPolicy;
        this.forceStreamAttach = forceStreamAttach;
        this.timeComputation = timeComputation;
        this.numberOfGPUs = numberOfGPUs;
    }

    @Override
    public String toString() {
        return "GrCUDATestOptionsStruct{" +
                "policy=" + policy +
                ", inputPrefetch=" + inputPrefetch +
                ", retrieveNewStreamPolicy=" + retrieveNewStreamPolicy +
                ", retrieveParentStreamPolicy=" + retrieveParentStreamPolicy +
                ", dependencyPolicy=" + dependencyPolicy +
                ", deviceSelectionPolicy=" + deviceSelectionPolicy +
                ", forceStreamAttach=" + forceStreamAttach +
                ", timeComputation=" + timeComputation +
                ", numberOfGPUs=" + numberOfGPUs +
                '}';
    }
}
