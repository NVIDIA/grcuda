package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.CUDAEvent;
import com.nvidia.grcuda.ParameterWithValue;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyComputation;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.DefaultStream;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

/**
 * Basic class that represents GrCUDA computations,
 * and is used to model data dependencies between computations;
 */
public abstract class GrCUDAComputationalElement {

    /**
     * This list contains the original set of input arguments that are used to compute dependencies;
     */
    protected final List<ParameterWithValue> argumentList;
    /**
     * Reference to the execution context where this computation is executed;
     */
    protected final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    /**
     * Reference to the stream where this computation will be executed,
     * if possible (i.e. if the computation can be executed on a custom stream).
     * Subclasses can keep an internal reference to the stream, e.g. if it can be manually modified by the user,
     * but it is required to keep that value consistent to this one if it is modified;
     */
    private CUDAStream stream = DefaultStream.get();
    /**
     * Reference to the event associated to this computation, and recorded on the stream where this computation is executed,
     * after the computation is started. It offers a precise synchronization point for children computations.
     * If the computation is not executed on a stream, the event is null;
     */
    private CUDAEvent event;
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
     * Specify if this computational element represents an array access (read or write) on an {@link com.nvidia.grcuda.array.AbstractArray}
     * performed synchronously by the CPU. By default it returns false;
     */
    protected boolean isComputationArrayAccess = false;

    private final DependencyComputation dependencyComputation;

    /**
     * Constructor that takes an argument set initializer to build the set of arguments used in the dependency computation
     * @param grCUDAExecutionContext execution context in which this computational element will be scheduled
     * @param initializer the initializer used to build the internal set of arguments considered in the dependency computation
     */
    public GrCUDAComputationalElement(AbstractGrCUDAExecutionContext grCUDAExecutionContext, InitializeArgumentList initializer) {
        this.argumentList = initializer.initialize();
        // Initialize by making a copy of the original set;
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.dependencyComputation = grCUDAExecutionContext.getDependencyBuilder().initialize(this.argumentList);
    }

    /**
     * Simplified constructor that takes a list of arguments, and consider all of them in the dependency computation
     * @param grCUDAExecutionContext execution context in which this computational element will be scheduled
     * @param args the list of arguments provided to the computation. Arguments are expected to be {@link org.graalvm.polyglot.Value}
     */
    public GrCUDAComputationalElement(AbstractGrCUDAExecutionContext grCUDAExecutionContext, List<ParameterWithValue> args) {
        this(grCUDAExecutionContext, new DefaultExecutionInitializer(args));
    }

    public List<ParameterWithValue> getArgumentList() {
        return argumentList;
    }

    /**
     * Return if this computation could lead to dependencies with future computations.
     * If not, this usually means that all of its arguments have already been superseded by other computations,
     * or that the computation didn't have any arguments to begin with;
     * @return if the computation could lead to future dependencies
     */
    public boolean hasPossibleDependencies() {
        return !this.dependencyComputation.getActiveArgumentSet().isEmpty();
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

    public Optional<CUDAEvent> getEvent() {
        if (event != null) {
            return Optional.of(event);
        } else {
            return Optional.empty();
        }
    }

    public void setEvent(CUDAEvent event) {
        this.event = event;
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
     * Retrieve how the dependency computations are computed;
     */
    public DependencyComputation getDependencyComputation() { return dependencyComputation; }

    /**
     * Set for all the {@link com.nvidia.grcuda.array.AbstractArray} in the computation if this computation is an array access;
     */
    public void updateIsComputationArrayAccess() {
        for (ParameterWithValue o : this.argumentList) {
            if (o.getArgumentValue() instanceof AbstractArray) {
                ((AbstractArray) o.getArgumentValue()).setLastComputationArrayAccess(isComputationArrayAccess);
            }
        }
    }

    /**
     * Computes if the "other" GrCUDAComputationalElement has dependencies w.r.t. this kernel,
     * such as requiring as input a value computed by this kernel;
     * @param other kernel for which we want to check dependencies, w.r.t. this kernel
     * @return the list of arguments that the two kernels have in common
     */
    public Collection<ParameterWithValue> computeDependencies(GrCUDAComputationalElement other) {
        return this.dependencyComputation.computeDependencies(other);
    }

    /**
     * Compute and return an additional stream dependency used by this computation.
     * This function is used by {@link com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager} to synchronize streams
     * that might not be directly used by this computation, but that have to be synchronized for this computation
     * to take place correctly. For example, in pre-Pascal GPUs it is required to ensure that no kernel is running if
     * the array accessed is visible to the global stream.
     * The actual invocation is wrapped by a {@link ArrayStreamArchitecturePolicy},
     * as the invocation depends on the GPU architecture;
     * @return An additional stream to synchronize
     */
    public final Optional<CUDAStream> additionalStreamDependency() {
        return grCUDAExecutionContext.getArrayStreamArchitecturePolicy().execute(this::additionalStreamDependencyImpl);
    }

    /**
     * Actual implementation of {@link GrCUDAComputationalElement#additionalStreamDependency}, it can be overridden
     * by concrete computations to provide additional streams for synchronization;
     * @return An additional stream to synchronize
     */
    protected Optional<CUDAStream> additionalStreamDependencyImpl() {
        return Optional.empty();
    }

    /**
     * The default initializer will simply store all the arguments,
     * and consider each of them in the dependency computations;
     */
    private static class DefaultExecutionInitializer implements InitializeArgumentList {
        private final List<ParameterWithValue> args;

        DefaultExecutionInitializer(List<ParameterWithValue> args) {
            this.args = args;
        }

        @Override
        public List<ParameterWithValue> initialize() {
            return args;
        }
    }
}
