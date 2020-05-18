package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.computation.ArrayStreamArchitecturePolicy;
import com.nvidia.grcuda.gpu.computation.PrePascalArrayStreamAssociation;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;

/**
 * Mock class to test the GrCUDAExecutionContextTest, it has a null CUDARuntime;
 */
public class GrCUDAExecutionContextMock extends GrCUDAExecutionContext {

    public GrCUDAExecutionContextMock(boolean syncStream) {
        super(null, null,
                new GrCUDAStreamManagerMock(null, syncStream), DependencyPolicyEnum.DEFAULT);
    }

    public GrCUDAExecutionContextMock() {
        super(null, null,
                new GrCUDAStreamManagerMock(null), DependencyPolicyEnum.DEFAULT);
    }

    public GrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy) {
        super(null, null,
                new GrCUDAStreamManagerMock(null), dependencyPolicy);
    }

    public GrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy, boolean syncStream, RetrieveNewStreamPolicyEnum retrieveStreamPolicy) {
        super(null, null,
                new GrCUDAStreamManagerMock(null, syncStream, retrieveStreamPolicy), dependencyPolicy);
    }

    public ArrayStreamArchitecturePolicy getArrayStreamArchitecturePolicy() {
        return new PrePascalArrayStreamAssociation();
    }
}
