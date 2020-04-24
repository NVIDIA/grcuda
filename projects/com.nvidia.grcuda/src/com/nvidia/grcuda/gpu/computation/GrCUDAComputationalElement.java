package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.gpu.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.DefaultStream;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

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
     * Reference to the stream where this computation will be executed,
     * if possible (i.e. if the computation can be executed on a custom stream).
     * Subclasses can keep an internal reference to the stream, e.g. if it can be manually modified by the user,
     * but it is required to keep that value consistent to this one if it is modified;
     */
    private CUDAStream stream = new DefaultStream();
    /**
     * Keep track of whether this computation has already been executed, and represents a "dead" vertex in the DAG.
     * Computations that are already executed will not be considered when computing dependencies;
     */
    private boolean computationFinished = false;
    /**
     * Keep track of whether this computation has already been started, to avoid performing the same computation multiple times;
     */
    private boolean computationStarted = false;

    /**
     * Constructor that takes an argument set initializer to build the set of arguments used in the dependency computation
     * @param grCUDAExecutionContext execution context in which this computational element will be scheduled
     * @param initializer the initializer used to build the internal set of arguments considered in the dependency computation
     */
    @CompilerDirectives.TruffleBoundary
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
    @CompilerDirectives.TruffleBoundary
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
    public Object schedule() throws UnsupportedTypeException {
        return this.grCUDAExecutionContext.registerExecution(this);
    }

    /**
     * Generic interface to perform the execution of this {@link GrCUDAComputationalElement}.
     * The actual execution implementation must be added by concrete computational elements.
     * The execution request will be done by the {@link GrCUDAExecutionContext}, after this computation has been scheduled
     * using {@link GrCUDAComputationalElement#schedule()}
     */
    public abstract Object execute() throws UnsupportedTypeException;

    public CUDAStream getStream() {
        return stream;
    }

    public void setStream(CUDAStream stream) {
        this.stream = stream;
    }

    public boolean isComputationFinished() {
        return computationFinished;
    }

    public boolean isComputationStarted() {
        return computationStarted;
    }

    public void setComputationFinished() {
        this.computationFinished = true;
    }

    public void setComputationStarted() {
        this.computationStarted = true;
    }

    /**
     * Find whether this computation should be done on a user-specified {@link com.nvidia.grcuda.gpu.stream.CUDAStream};
     * If not, the stream will be provided internally using the specified execution policy. By default return false;
     * @return if the computation is done on a custom CUDA stream;
     */
    public boolean useManuallySpecifiedStream() { return false; }

    /**
     * Some computational elements, like kernels, can be executed on different {@link CUDAStream} to provide
     * parallel asynchronous execution. Other computations, such as array reads, do not require streams, or cannot be
     * executed on streams different from the {@link DefaultStream};
     * @return if this computation can be executed on a customized stream
     */
    public boolean canUseStream() { return false; }

    /**
     * Provide a way to associate input arrays allocated using managed memory to the stream
     * on which this kernel is executed. This is required by pre-Pascal GPUs to allow the CPU to access
     * managed memory belonging to arrays not used by kernels running on the GPU.
     * By default, the implementation is empty, as {@link GrCUDAComputationalElement#canUseStream} is false;
     */
    public final void associateArraysToStream() {
        grCUDAExecutionContext.getArrayStreamArchitecturePolicy().execute(this::associateArraysToStreamImpl);
    }

    /**
     * Actual implementation of {@link GrCUDAComputationalElement#associateArraysToStream()},
     * to be modified by concrete computational elements;
     */
    protected void associateArraysToStreamImpl() {}

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
