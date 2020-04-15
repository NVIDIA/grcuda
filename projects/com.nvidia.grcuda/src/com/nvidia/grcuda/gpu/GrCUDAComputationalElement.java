package com.nvidia.grcuda.gpu;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Basic class that represents GrCUDA computations,
 * and is used to model data dependencies between computations;
 */
public abstract class GrCUDAComputationalElement {

    /**
     * This set contains the input arguments that are used to compute dependencies;
     */
    protected final Set<Object> argumentSet;
    /**
     * Reference to the execution context where this computation is executed;
     */
    protected final GrCUDAExecutionContext grCUDAExecutionContext;

    /**
     * Constructor that takes an argument set initializer to build the set of arguments used in the dependency computation
     * @param grCUDAExecutionContext execution context in which this computational element will be scheduled
     * @param initializer the initializer used to build the internal set of arguments considered in the dependency computation
     */
    public GrCUDAComputationalElement(GrCUDAExecutionContext grCUDAExecutionContext, InitializeArgumentSet initializer) {
        this.argumentSet = initializer.initialize();
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.grCUDAExecutionContext.registerExecution(this);
    }

    /**
     * Simplified constructor that takes a list of arguments, and consider all of them in the dependency computation
     * @param grCUDAExecutionContext execution context in which this computational element will be scheduled
     * @param args the list of arguments provided to the computation. Arguments are expected to be {@link org.graalvm.polyglot.Value}
     */
    public GrCUDAComputationalElement(GrCUDAExecutionContext grCUDAExecutionContext, List<Object> args) {
        this.argumentSet = new DefaultExecutionInitializer(args).initialize();
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.grCUDAExecutionContext.registerExecution(this);
    }

    public Set<Object> getArgumentSet() {
        return argumentSet;
    }

    /**
     * Computes if the "other" GrCUDAComputationalElement has dependencies w.r.t. this kernel,
     * such as requiring as input a value computed by this kernel;
     * @param other kernel for which we want to check dependencies, w.r.t. this kernel
     * @return the list of arguments that the two kernels have in common
     */
    public List<Object> computeDependencies(GrCUDAComputationalElement other) {
        // Obtain the common dependencies through set intersection;
        Set<Object> intersection = new HashSet<>(argumentSet);
        intersection.retainAll(other.argumentSet);
        return new ArrayList<>(intersection);
    }

    /**
     * The default initializer will simply store all the arguments,
     * and consider each of them in the dependency computations;
     */
    private static class DefaultExecutionInitializer implements InitializeArgumentSet {
        private final List<Object> args;

        DefaultExecutionInitializer(List<Object> args) {
            this.args = args;
        }

        @Override
        public Set<Object> initialize() {
            return new HashSet<>(args);
        }
    }
}
