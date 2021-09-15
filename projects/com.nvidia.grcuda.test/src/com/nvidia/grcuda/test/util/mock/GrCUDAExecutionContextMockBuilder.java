package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveParentStreamPolicyEnum;

public class GrCUDAExecutionContextMockBuilder {

    DependencyPolicyEnum dependencyPolicy = DependencyPolicyEnum.NO_CONST;
    RetrieveNewStreamPolicyEnum retrieveStreamPolicy = RetrieveNewStreamPolicyEnum.FIFO;
    RetrieveParentStreamPolicyEnum parentStreamPolicyEnum = RetrieveParentStreamPolicyEnum.SAME_AS_PARENT;

    public GrCUDAExecutionContextMock build() {
        return new GrCUDAExecutionContextMock(dependencyPolicy, retrieveStreamPolicy, parentStreamPolicyEnum);
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
