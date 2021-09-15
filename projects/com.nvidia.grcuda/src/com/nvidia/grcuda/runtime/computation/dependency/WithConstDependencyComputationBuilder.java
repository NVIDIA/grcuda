package com.nvidia.grcuda.runtime.computation.dependency;

import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;

import java.util.List;

public class WithConstDependencyComputationBuilder implements DependencyComputationBuilder {

    @Override
    public WithConstDependencyComputation initialize(List<ComputationArgumentWithValue> argumentList) {
        return new WithConstDependencyComputation(argumentList);
    }
}
