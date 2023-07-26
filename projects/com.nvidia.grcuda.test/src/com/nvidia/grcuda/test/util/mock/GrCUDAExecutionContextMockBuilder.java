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
package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;

public class GrCUDAExecutionContextMockBuilder {

    DependencyPolicyEnum dependencyPolicy = DependencyPolicyEnum.NO_CONST;
    RetrieveNewStreamPolicyEnum retrieveStreamPolicy = RetrieveNewStreamPolicyEnum.REUSE;
    RetrieveParentStreamPolicyEnum parentStreamPolicyEnum = RetrieveParentStreamPolicyEnum.SAME_AS_PARENT;
    DeviceSelectionPolicyEnum deviceSelectionPolicyEnum = DeviceSelectionPolicyEnum.SINGLE_GPU;
    boolean isArchitecturePascalOrNewer = true;
    int numberOfAvailableGPUs = 1;
    int numberOfGPUsToUse = 1;

    public AsyncGrCUDAExecutionContextMock build() {
        return new AsyncGrCUDAExecutionContextMock(dependencyPolicy, retrieveStreamPolicy, parentStreamPolicyEnum, deviceSelectionPolicyEnum, isArchitecturePascalOrNewer, numberOfAvailableGPUs, numberOfGPUsToUse);
    }

    public GrCUDAExecutionContextMockBuilder setDependencyPolicy(DependencyPolicyEnum dependencyPolicy) {
        this.dependencyPolicy = dependencyPolicy;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setRetrieveNewStreamPolicy(RetrieveNewStreamPolicyEnum retrieveStreamPolicy) {
        this.retrieveStreamPolicy = retrieveStreamPolicy;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setDeviceSelectionPolicy(DeviceSelectionPolicyEnum deviceSelectionPolicyEnum) {
        this.deviceSelectionPolicyEnum = deviceSelectionPolicyEnum;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum retrieveStreamPolicy) {
        this.parentStreamPolicyEnum = retrieveStreamPolicy;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setArchitecturePascalOrNewer(boolean isArchitecturePascalOrNewer) {
        this.isArchitecturePascalOrNewer = isArchitecturePascalOrNewer;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setNumberOfAvailableGPUs(int numberOfAvailableGPUs) {
        this.numberOfAvailableGPUs = numberOfAvailableGPUs;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setNumberOfGPUsToUse(int numberOfGPUsToUse) {
        this.numberOfGPUsToUse = numberOfGPUsToUse;
        return this;
    }
}
