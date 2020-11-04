package com.nvidia.grcuda.gpu.computation.dependency;

import com.nvidia.grcuda.ComputationArgumentWithValue;

import java.util.List;

public interface DependencyComputationBuilder {
    DependencyComputation initialize(List<ComputationArgumentWithValue> argumentList);
}
