package com.nvidia.grcuda.gpu.computation;

import java.util.Set;

public interface InitializeArgumentSet {
    /**
     * Used by different {@link GrCUDAComputationalElement} to initialize the set of arguments
     * considered in the dependency evaluation.
     * @return a set of arguments used in the evaluation
     */
    Set<ComputationArgumentWithValue> initialize();
}
