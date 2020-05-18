package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;

public class GrCUDAExecutionContextMockBuilder {

    boolean syncStream = false;
    DependencyPolicyEnum dependencyPolicy = DependencyPolicyEnum.DEFAULT;
    RetrieveNewStreamPolicyEnum retrieveStreamPolicy = RetrieveNewStreamPolicyEnum.FIFO;

    public GrCUDAExecutionContextMock build() {
        return new GrCUDAExecutionContextMock(dependencyPolicy, syncStream, retrieveStreamPolicy);
    }

    public GrCUDAExecutionContextMockBuilder setSyncStream(boolean syncStream) {
        this.syncStream = syncStream;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setDependencyPolicy(DependencyPolicyEnum dependencyPolicy) {
        this.dependencyPolicy = dependencyPolicy;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setRetrieveStreamPolicy(RetrieveNewStreamPolicyEnum retrieveStreamPolicy) {
        this.retrieveStreamPolicy = retrieveStreamPolicy;
        return this;
    }
}
