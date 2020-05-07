package com.nvidia.grcuda.gpu.computation.dependency;

import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;

import java.util.ArrayList;
import java.util.List;

/**
 * If two computations have the same argument, but it is read-only in both cases (i.e. const),
 * there is no reason to create a dependency between the two ;
 */
public class WithConstDependencyComputation extends DependencyComputation {

    WithConstDependencyComputation(List<ComputationArgumentWithValue> argumentList) {
        activeArgumentSet = new ArrayList<>(argumentList);
    }

    @Override
    public List<ComputationArgumentWithValue> computeDependencies(GrCUDAComputationalElement other) {
        List<ComputationArgumentWithValue> dependencies = new ArrayList<>();
        List<ComputationArgumentWithValue> newArgumentSet = new ArrayList<>();
        for (ComputationArgumentWithValue arg : activeArgumentSet) {
            boolean dependencyFound = false;
            for (ComputationArgumentWithValue otherArg : other.getDependencyComputation().getActiveArgumentSet()) {
                // If both arguments are const, we skip the dependency;
                if (arg.equals(otherArg) && !(arg.isConst() && otherArg.isConst())) {
                    dependencies.add(arg);
                    dependencyFound = true;
                    // If the other argument is const, the current argument must be added to newArgumentSet
                    //   as it could cause other dependencies in the future;
                    if (otherArg.isConst()) {
                        newArgumentSet.add(arg);
                    }
                    break;
                }
            }
            if (!dependencyFound) {
                // Otherwise, the current argument is still "active", and could enforce a dependency on a future computation;
                newArgumentSet.add(arg);
            }
        }
        // Arguments that are not leading to a new dependency could still create new dependencies later on!
        activeArgumentSet = newArgumentSet;
        // Return the list of arguments that created dependencies with the new computation;
        return dependencies;
    }

    @Override
    public boolean keepArgument(ComputationArgumentWithValue arg) {
        return (arg.getArgumentValue() instanceof AbstractArray) && !arg.isConst();
    }
}
