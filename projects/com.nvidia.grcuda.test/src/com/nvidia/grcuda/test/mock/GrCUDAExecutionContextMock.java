package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.computation.ArrayStreamArchitecturePolicy;
import com.nvidia.grcuda.gpu.computation.PrePascalArrayStreamAssociation;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.computation.prefetch.PrefetcherEnum;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;

/**
 * Mock class to test the GrCUDAExecutionContextTest, it has a null CUDARuntime;
 */
public class GrCUDAExecutionContextMock extends GrCUDAExecutionContext {

    public GrCUDAExecutionContextMock() {
        super(null, null,
                new GrCUDAStreamManagerMock(null), DependencyPolicyEnum.NO_CONST, PrefetcherEnum.NONE);
    }

    public GrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy) {
        super(null, null,
                new GrCUDAStreamManagerMock(null), dependencyPolicy, PrefetcherEnum.NONE);
    }

    public GrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy,
                                      RetrieveNewStreamPolicyEnum retrieveStreamPolicy,
                                      RetrieveParentStreamPolicyEnum parentStreamPolicyEnum) {
        super(null, null,
                new GrCUDAStreamManagerMock(null, retrieveStreamPolicy, parentStreamPolicyEnum), dependencyPolicy, PrefetcherEnum.NONE);
    }

    public ArrayStreamArchitecturePolicy getArrayStreamArchitecturePolicy() {
        return new PrePascalArrayStreamAssociation();
    }
}
