package com.nvidia.grcuda.gpu.computation;

import java.util.List;

public interface InitializeArgumentList {
    /**
     * Used by different {@link GrCUDAComputationalElement} to initialize the list of arguments
     * considered in the dependency evaluation.
     * @return a list of arguments used in the evaluation
     */
    List<ComputationArgumentWithValue> initialize();
}
