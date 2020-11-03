package com.nvidia.grcuda.gpu.computation.dependency;

import com.nvidia.grcuda.ParameterWithValue;

import java.util.List;

public class DefaultDependencyComputationBuilder implements DependencyComputationBuilder {

    @Override
    public DefaultDependencyComputation initialize(List<ParameterWithValue> argumentList) {
        return new DefaultDependencyComputation(argumentList);
    }
}
