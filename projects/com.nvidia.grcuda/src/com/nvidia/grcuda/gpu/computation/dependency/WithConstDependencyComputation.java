package com.nvidia.grcuda.gpu.computation.dependency;

import com.nvidia.grcuda.ParameterWithValue;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;

import java.util.ArrayList;
import java.util.List;

/**
 * If two computations have the same argument, but it is read-only in both cases (i.e. const),
 * there is no reason to create a dependency between the two ;
 */
public class WithConstDependencyComputation extends DependencyComputation {

    WithConstDependencyComputation(List<ParameterWithValue> argumentList) {
        activeArgumentSet = new ArrayList<>(argumentList);
    }

    @Override
    public List<ParameterWithValue> computeDependencies(GrCUDAComputationalElement other) {
        List<ParameterWithValue> dependencies = new ArrayList<>();
        List<ParameterWithValue> newArgumentSet = new ArrayList<>();
        for (ParameterWithValue arg : activeArgumentSet) {
            boolean dependencyFound = false;
            for (ParameterWithValue otherArg : other.getDependencyComputation().getActiveArgumentSet()) {
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

    /**
     * If the array was attached to a stream, and now it is a const parameter, reset its visibility to the default stream.
     * For simplicity, we keep the visibility of all arguments currently used as const to the default stream.
     * This allow the scheduling of multiple computations that use the same argument as const;
     * @param arg an argument to analyse
     * @return if this argument visibility should be reset or not
     */
    @Override
    public boolean streamResetAttachFilter(ParameterWithValue arg) {
        return arg.isConst();
    }
}
