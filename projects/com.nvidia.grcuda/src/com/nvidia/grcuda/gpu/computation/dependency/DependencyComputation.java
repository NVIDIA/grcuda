package com.nvidia.grcuda.gpu.computation.dependency;

import com.nvidia.grcuda.gpu.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;

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

    /**
     * Provide an additional, optional filter used by dependency computations.
     * The filter defines if and how arguments can be used in user-defined {@link GrCUDAComputationalElement}
     * whose behaviour is influenced by how dependencies are computed.
     * For example, a filter could keep only non-const array arguments, and ignore the others;
     * @param arg an argument to analyse
     * @return if this argument should be kept or not for the user-defined computation
     */
    public boolean keepArgument(ComputationArgumentWithValue arg) {
        return true;
    }
}
