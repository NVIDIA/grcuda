package com.nvidia.grcuda.gpu.computation.dependency;

import com.nvidia.grcuda.ComputationArgumentWithValue;

import java.util.List;

public class DefaultDependencyComputationBuilder implements DependencyComputationBuilder {

    @Override
    public DefaultDependencyComputation initialize(List<ComputationArgumentWithValue> argumentList) {
        return new DefaultDependencyComputation(argumentList);
    }
}
