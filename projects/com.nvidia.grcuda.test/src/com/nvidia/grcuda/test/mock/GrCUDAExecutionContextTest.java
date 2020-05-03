package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.computation.ArrayStreamArchitecturePolicy;
import com.nvidia.grcuda.gpu.computation.PrePascalArrayStreamAssociation;
import com.nvidia.grcuda.gpu.computation.dependency.DefaultDependencyComputationBuilder;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyComputationBuilder;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;

/**
 * Mock class to test the GrCUDAExecutionContextTest, it has a null CUDARuntime;
 */
public class GrCUDAExecutionContextTest extends GrCUDAExecutionContext {

    public GrCUDAExecutionContextTest(boolean syncStream) {
        super(null, null,
                new GrCUDAStreamManagerTest(null, syncStream), new DefaultDependencyComputationBuilder());
    }

    public GrCUDAExecutionContextTest() {
        super(null, null,
                new GrCUDAStreamManagerTest(null), new DefaultDependencyComputationBuilder());
    }

    public GrCUDAExecutionContextTest(DependencyComputationBuilder dependencyBuilder) {
        super(null, null,
                new GrCUDAStreamManagerTest(null), dependencyBuilder);
    }

    public GrCUDAExecutionContextTest(DependencyComputationBuilder dependencyBuilder, boolean syncStream) {
        super(null, null,
                new GrCUDAStreamManagerTest(null, syncStream), dependencyBuilder);
    }

    public ArrayStreamArchitecturePolicy getArrayStreamArchitecturePolicy() {
        return new PrePascalArrayStreamAssociation();
    }
}
