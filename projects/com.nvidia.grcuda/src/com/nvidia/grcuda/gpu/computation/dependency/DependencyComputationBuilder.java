package com.nvidia.grcuda.gpu.computation.dependency;

import com.nvidia.grcuda.ParameterWithValue;

import java.util.List;

public interface DependencyComputationBuilder {
    DependencyComputation initialize(List<ParameterWithValue> argumentList);
}
