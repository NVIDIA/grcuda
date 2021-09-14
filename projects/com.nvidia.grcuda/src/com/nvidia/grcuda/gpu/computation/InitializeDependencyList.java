package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.ComputationArgumentWithValue;

import java.util.List;

public interface InitializeDependencyList {
    /**
     * Used by different {@link GrCUDAComputationalElement} to initialize the list of arguments
     * considered in the dependency evaluation.
     * @return a list of arguments used in the dependency evaluation
     */
    List<ComputationArgumentWithValue> initialize();
}
