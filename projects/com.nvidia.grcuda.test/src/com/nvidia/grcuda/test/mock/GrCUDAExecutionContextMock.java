package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.computation.ArrayStreamArchitecturePolicy;
import com.nvidia.grcuda.gpu.computation.PrePascalArrayStreamAssociation;
import com.nvidia.grcuda.gpu.computation.dependency.DefaultDependencyComputationBuilder;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyComputationBuilder;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.RetrieveStreamPolicyEnum;

/**
 * Mock class to test the GrCUDAExecutionContextTest, it has a null CUDARuntime;
 */
public class GrCUDAExecutionContextMock extends GrCUDAExecutionContext {

    public GrCUDAExecutionContextMock(boolean syncStream) {
        super(null, null,
                new GrCUDAStreamManagerMock(null, syncStream), new DefaultDependencyComputationBuilder());
    }

    public GrCUDAExecutionContextMock() {
        super(null, null,
                new GrCUDAStreamManagerMock(null), new DefaultDependencyComputationBuilder());
    }

    public GrCUDAExecutionContextMock(DependencyComputationBuilder dependencyBuilder) {
        super(null, null,
                new GrCUDAStreamManagerMock(null), dependencyBuilder);
    }

    public GrCUDAExecutionContextMock(DependencyComputationBuilder dependencyBuilder, boolean syncStream, RetrieveStreamPolicyEnum retrieveStreamPolicy) {
        super(null, null,
                new GrCUDAStreamManagerMock(null, syncStream, retrieveStreamPolicy), dependencyBuilder);
    }

    public ArrayStreamArchitecturePolicy getArrayStreamArchitecturePolicy() {
        return new PrePascalArrayStreamAssociation();
    }
}
