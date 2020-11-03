package com.nvidia.grcuda.gpu.computation.dependency;

import com.nvidia.grcuda.ParameterWithValue;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.computation.InitializeArgumentList;
import com.oracle.truffle.api.CompilerDirectives;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * By default, consider all dependencies in the active argument set,
 * initially specified by the {@link InitializeArgumentList} interface.
 * Also update the active argument set, by adding all arguments that were not included in a dependency relation;
 */
public class DefaultDependencyComputation extends DependencyComputation {

    @CompilerDirectives.TruffleBoundary
    DefaultDependencyComputation(List<ParameterWithValue> argumentList) {
        activeArgumentSet = new HashSet<>(argumentList);
    }

    static DefaultDependencyComputation initialize(List<ParameterWithValue> argumentList) {
        return new DefaultDependencyComputation(argumentList);
    }

    @CompilerDirectives.TruffleBoundary
    @Override
    public List<ParameterWithValue> computeDependencies(GrCUDAComputationalElement other) {
        Set<ParameterWithValue> dependencies = new HashSet<>();
        Set<ParameterWithValue> newArgumentSet = new HashSet<>();
        for (ParameterWithValue arg : activeArgumentSet) {
            // The other computation requires the current argument, so we have found a new dependency;
            if (other.getDependencyComputation().getActiveArgumentSet().contains(arg)) {
                dependencies.add(arg);
            } else {
                // Otherwise, the current argument is still "active", and could enforce a dependency on a future computation;
                newArgumentSet.add(arg);
            }
        }
        // Arguments that are not leading to a new dependency could still create new dependencies later on!
        activeArgumentSet = newArgumentSet;
        // Return the list of arguments that created dependencies with the new computation;
        return new ArrayList<>(dependencies);
    }
}
