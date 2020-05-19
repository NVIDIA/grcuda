package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;

public class GrCUDAExecutionContextMockBuilder {

    boolean syncStream = false;
    DependencyPolicyEnum dependencyPolicy = DependencyPolicyEnum.DEFAULT;
    RetrieveNewStreamPolicyEnum retrieveStreamPolicy = RetrieveNewStreamPolicyEnum.FIFO;
    RetrieveParentStreamPolicyEnum parentStreamPolicyEnum = RetrieveParentStreamPolicyEnum.DEFAULT;

    public GrCUDAExecutionContextMock build() {
        return new GrCUDAExecutionContextMock(dependencyPolicy, syncStream, retrieveStreamPolicy, parentStreamPolicyEnum);
    }

    public GrCUDAExecutionContextMockBuilder setSyncStream(boolean syncStream) {
        this.syncStream = syncStream;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setDependencyPolicy(DependencyPolicyEnum dependencyPolicy) {
        this.dependencyPolicy = dependencyPolicy;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setRetrieveNewStreamPolicy(RetrieveNewStreamPolicyEnum retrieveStreamPolicy) {
        this.retrieveStreamPolicy = retrieveStreamPolicy;
        return this;
    }

    public GrCUDAExecutionContextMockBuilder setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum retrieveStreamPolicy) {
        this.parentStreamPolicyEnum = retrieveStreamPolicy;
        return this;
    }
}
