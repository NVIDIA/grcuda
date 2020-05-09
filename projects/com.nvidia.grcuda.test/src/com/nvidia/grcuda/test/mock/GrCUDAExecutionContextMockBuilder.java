package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveStreamPolicyEnum;

public class GrCUDAExecutionContextMockBuilder {

    boolean syncStream = false;
    DependencyPolicyEnum dependencyPolicy = DependencyPolicyEnum.DEFAULT;
    RetrieveStreamPolicyEnum retrieveStreamPolicy = RetrieveStreamPolicyEnum.LIFO;

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

    public GrCUDAExecutionContextMockBuilder setRetrieveStreamPolicy(RetrieveStreamPolicyEnum retrieveStreamPolicy) {
        this.retrieveStreamPolicy = retrieveStreamPolicy;
        return this;
    }
}
