package com.nvidia.grcuda.gpu.executioncontext;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.Kernel;
import com.nvidia.grcuda.gpu.computation.ArrayStreamArchitecturePolicy;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.computation.dependency.DefaultDependencyComputationBuilder;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyComputationBuilder;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.computation.dependency.WithConstDependencyComputationBuilder;
import com.nvidia.grcuda.gpu.computation.prefetch.AbstractArrayPrefetcher;
import com.nvidia.grcuda.gpu.computation.prefetch.DefaultArrayPrefetcher;
import com.nvidia.grcuda.gpu.computation.prefetch.NoneArrayPrefetcher;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.HashSet;
import java.util.Set;

/**
 * Abstract class that defines how {@link GrCUDAComputationalElement} are registered and scheduled for execution.
 * It monitor sthe state of GrCUDA execution, keep track of memory allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public abstract class AbstractGrCUDAExecutionContext {

    /**
     * Reference to the inner {@link CUDARuntime} used to execute kernels and other {@link GrCUDAComputationalElement}
     */
    protected final CUDARuntime cudaRuntime;

    /**
     * Set that contains all the arrays allocated so far.
     */
    protected final Set<AbstractArray> arraySet = new HashSet<>();

    /**
     * Set that contains all the CUDA kernels declared so far.
     */
    protected final Set<Kernel> kernelSet = new HashSet<>();

    /**
     * Reference to the computational DAG that represents dependencies between computations;
     */
    protected final ExecutionDAG dag;

    /**
     * Reference to how dependencies between computational elements are computed within this execution context;
     */
    private final DependencyComputationBuilder dependencyBuilder;

    /**
     * Reference to the prefetching strategy to use in this execution context;
     */
    protected final AbstractArrayPrefetcher arrayPrefetcher;

    public AbstractGrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, DependencyPolicyEnum dependencyPolicy) {
        this(new CUDARuntime(context, env), dependencyPolicy, false);
    }

    public AbstractGrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, DependencyPolicyEnum dependencyPolicy, boolean inputPrefetch) {
        this(new CUDARuntime(context, env), dependencyPolicy, inputPrefetch);
    }

    public AbstractGrCUDAExecutionContext(CUDARuntime cudaRuntime, DependencyPolicyEnum dependencyPolicy, boolean inputPrefetch) {
        this.cudaRuntime = cudaRuntime;
        // Compute the dependency policy to use;
        switch (dependencyPolicy) {
            case WITH_CONST:
                this.dependencyBuilder = new WithConstDependencyComputationBuilder();
                break;
            case DEFAULT:
                this.dependencyBuilder = new DefaultDependencyComputationBuilder();
                break;
            default:
                this.dependencyBuilder = new DefaultDependencyComputationBuilder();
        }
        // Compute the prefetcher to use;
        if (inputPrefetch && this.cudaRuntime.isArchitectureIsPascalOrNewer()) {
            arrayPrefetcher = new DefaultArrayPrefetcher(this.cudaRuntime);
        } else {
            arrayPrefetcher = new NoneArrayPrefetcher(this.cudaRuntime);
        }
        this.dag = new ExecutionDAG(dependencyPolicy);
    }

    /**
     * Register this computation for future execution by the {@link AbstractGrCUDAExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data dependencies;
     */
    abstract public Object registerExecution(GrCUDAComputationalElement computation) throws UnsupportedTypeException;

    public void registerArray(AbstractArray array) {
        arraySet.add(array);
    }

    public void registerKernel(Kernel kernel) {
        kernelSet.add(kernel);
    }

    public ExecutionDAG getDag() {
        return dag;
    }

    public CUDARuntime getCudaRuntime() {
        return cudaRuntime;
    }

    public DependencyComputationBuilder getDependencyBuilder() {
        return dependencyBuilder;
    }

    // Functions used to interface directly with the CUDA runtime;

    public Kernel loadKernel(String cubinFile, String kernelName, String signature) {
        return cudaRuntime.loadKernel(this, cubinFile, kernelName, signature);
    }

    public Kernel buildKernel(String code, String kernelName, String signature) {
        return cudaRuntime.buildKernel(this, code, kernelName, signature);
    }

    public ArrayStreamArchitecturePolicy getArrayStreamArchitecturePolicy() {
        return cudaRuntime.getArrayStreamArchitecturePolicy();
    }

    /**
     * Check if any computation is currently marked as active, and is running on a stream managed by this context.
     * If so, scheduling of new computations is likely to require synchronizations of some sort;
     * @return if any computation is considered active on a stream managed by this context
     */
    public abstract boolean isAnyComputationActive();

    /**
     * Delete internal structures that require manual cleanup operations;
     */
    public void cleanup() { }
}
