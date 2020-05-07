package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.computation.dependency.DefaultDependencyComputationBuilder;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyComputationBuilder;
import com.nvidia.grcuda.gpu.stream.RetrieveStreamPolicyEnum;

public class GrCUDAExecutionContextMockBuilder {

    boolean syncStream = false;
    DependencyComputationBuilder dependencyComputationBuilder = new DefaultDependencyComputationBuilder();
    RetrieveStreamPolicyEnum retrieveStreamPolicy = RetrieveStreamPolicyEnum.LIFO;

    public GrCUDAExecutionContextMock build() {
        return new GrCUDAExecutionContextMock(dependencyComputationBuilder, syncStream, retrieveStreamPolicy);
    }

    public GrCUDAExecutionContextMockBuilder setSyncStream(boolean syncStream) {
        this.syncStream = syncStream;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setDependencyComputationBuilder(DependencyComputationBuilder dependencyComputationBuilder) {
        this.dependencyComputationBuilder = dependencyComputationBuilder;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setRetrieveStreamPolicy(RetrieveStreamPolicyEnum retrieveStreamPolicy) {
        this.retrieveStreamPolicy = retrieveStreamPolicy;
        return this;
    }
}
