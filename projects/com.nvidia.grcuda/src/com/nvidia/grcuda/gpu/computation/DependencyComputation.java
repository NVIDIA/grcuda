package com.nvidia.grcuda.gpu.computation;

import java.util.Collection;

/**
 * Defines how data dependencies between {@link GrCUDAComputationalElement} are found,
 * e.g. if read-only or scalar argyments should be ignored.
 * It returns the list of arguments that have been found to create side-effects.
 * The function is not guaranteed to be pure,
 * and is allowed update information in the {@link GrCUDAComputationalElement}
 */
public abstract class DependencyComputation {

    /**
     * This set contains the input arguments that are considered, at each step, in the dependency computation.
     * The set initially coincides with "argumentSet", then arguments are removed from this set once a new dependency is found.
     * This is conceptually a set, in the sense that every element is unique.
     * Concrete implementations might use other data structures, if required;
     */
    protected Collection<ComputationArgumentWithValue> activeArgumentSet;

    /**
     * Computes if the "other" GrCUDAComputationalElement has dependencies w.r.t. this kernel,
     * such as requiring as input a value computed by this kernel;
     * @param other kernel for which we want to check dependencies, w.r.t. this kernel
     * @return the list of arguments that the two kernels have in common
     */
    public abstract Collection<ComputationArgumentWithValue> computeDependencies(GrCUDAComputationalElement other);

    public Collection<ComputationArgumentWithValue> getActiveArgumentSet() {
        return activeArgumentSet;
    }
}
