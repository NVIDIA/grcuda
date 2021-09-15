package com.nvidia.grcuda.runtime.computation.dependency;

import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;

import java.util.List;

public class DefaultDependencyComputationBuilder implements DependencyComputationBuilder {

    @Override
    public DefaultDependencyComputation initialize(List<ComputationArgumentWithValue> argumentList) {
        return new DefaultDependencyComputation(argumentList);
    }
}
