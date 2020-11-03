package com.nvidia.grcuda.gpu.computation.dependency;

import com.nvidia.grcuda.ParameterWithValue;

import java.util.List;

public class WithConstDependencyComputationBuilder implements DependencyComputationBuilder {

    @Override
    public WithConstDependencyComputation initialize(List<ParameterWithValue> argumentList) {
        return new WithConstDependencyComputation(argumentList);
    }
}
