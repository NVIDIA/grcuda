package com.nvidia.grcuda.gpu;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Basic class that represents GrCUDA computations,
 * and is used to model data dependencies between computations;
 */
public abstract class GrCUDAComputationalElement {

    /**
     * This set contains the original set of input arguments that are used to compute dependencies;
     */
    protected final Set<Object> argumentSet;
    /**
     * This set contains the input arguments that are considered, at each step, in the dependency computation.
     * The set initially coincides with "argumentSet", then arguments are removed from this set once a new dependency is found;
     * TODO: should this be moved somewhere else? e.g. inside the DAG, although this means moving the dependency computation too
     */
    private Set<Object> activeArgumentSet;
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
        // Initialize by making a copy of the original set;
        this.activeArgumentSet = new HashSet<>(this.argumentSet);
        this.grCUDAExecutionContext = grCUDAExecutionContext;
    }

    /**
     * Simplified constructor that takes a list of arguments, and consider all of them in the dependency computation
     * @param grCUDAExecutionContext execution context in which this computational element will be scheduled
     * @param args the list of arguments provided to the computation. Arguments are expected to be {@link org.graalvm.polyglot.Value}
     */
    public GrCUDAComputationalElement(GrCUDAExecutionContext grCUDAExecutionContext, List<Object> args) {
        this(grCUDAExecutionContext, new DefaultExecutionInitializer(args));
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
        Set<Object> dependencies = new HashSet<>();
        Set<Object> newArgumentSet = new HashSet<>();
        for (Object arg : this.activeArgumentSet) {
            // The other computation requires the current argument, so we have found a new dependency;
            if (other.activeArgumentSet.contains(arg)) {
                dependencies.add(arg);
            } else {
                // Otherwise, the current argument is still "active", and could enforce a dependency on a future computation;
                newArgumentSet.add(arg);
            }
        }
        // Arguments that are not leading to a new dependency could still create new dependencies later on!
        this.activeArgumentSet = newArgumentSet;
        // Return the list of arguments that created dependencies with the new computation;
        return new ArrayList<>(dependencies);
    }

    /**
     * Return if this computation could lead to dependencies with future computations.
     * If not, this usually means that all of its arguments have already been superseded by other computations,
     * or that the computation didn't have any arguments to begin with;
     * @return if the computation could lead to future dependencies
     */
    public boolean hasPossibleDependencies() {
        return !this.activeArgumentSet.isEmpty();
    }

    /**
     * Schedule this computation for future execution by the {@link GrCUDAExecutionContext}.
     * The scheduling request is separate from the {@link GrCUDAComputationalElement} instantiation
     * as we need to ensure that the the computational element subclass has been completely instantiated;
     */
    public void schedule() {
        this.grCUDAExecutionContext.registerExecution(this);
    }

    /**
     * Generic interface to perform the execution of this {@link GrCUDAComputationalElement}.
     * The actual execution implementation must be added by concrete computational elements.
     * The execution request will be done by the {@link GrCUDAExecutionContext}, after this computation has been scheduled
     * using {@link GrCUDAComputationalElement.schedule()}
     */
    public abstract void execute();

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
