package com.nvidia.grcuda.gpu;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Basic class that represents GrCUDA computations,
 * and is used to model data dependencies between computations
 */
public abstract class GrCUDAComputationalElement {

    /**
     * This set contains the input arguments that are used to compute dependencies;
     */
    protected final Set<Object> argumentSet;

    public GrCUDAComputationalElement(InitializeArgumentSet initializer) {
        this.argumentSet = initializer.initialize();
    }

    public GrCUDAComputationalElement() {
        this.argumentSet = new DefaultInitializer().initialize();
    }

    public Set<Object> getArgumentSet() {
        return argumentSet;
    }

    /**
     * Computes if the "other" GrCUDAComputationalElement has dependencies w.r.t. this kernel,
     * such as requiring as input a value computed by this kernel.
     * @param other kernel for which we want to check dependencies, w.r.t. this kernel
     * @return the list of arguments that the two kernels have in common
     */
    public List<Object> computeDependencies(GrCUDAComputationalElement other) {
        // Obtain the common dependencies through set intersection;
        Set<Object> intersection = new HashSet<>(argumentSet);
        intersection.retainAll(other.argumentSet);
        return new ArrayList<>(intersection);
    }

    private static class DefaultInitializer implements InitializeArgumentSet {
        @Override
        public Set<Object> initialize() {
            return new HashSet<>();
        }
    }
}
