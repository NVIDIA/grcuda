package com.nvidia.grcuda.gpu.computation.dependency;

import com.nvidia.grcuda.ComputationArgumentWithValue;

import java.util.List;

public class WithConstDependencyComputationBuilder implements DependencyComputationBuilder {

    @Override
    public WithConstDependencyComputation initialize(List<ComputationArgumentWithValue> argumentList) {
        return new WithConstDependencyComputation(argumentList);
    }
}
