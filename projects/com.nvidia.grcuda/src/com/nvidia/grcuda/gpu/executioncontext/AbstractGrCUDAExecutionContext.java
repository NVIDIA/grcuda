package com.nvidia.grcuda.gpu.executioncontext;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.Kernel;
import com.nvidia.grcuda.gpu.computation.ArrayStreamArchitecturePolicy;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
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
    protected final ExecutionDAG dag = new ExecutionDAG();

    public AbstractGrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env) {
        this.cudaRuntime = new CUDARuntime(context, env);
    }

    public AbstractGrCUDAExecutionContext(CUDARuntime cudaRuntime) {
        this.cudaRuntime = cudaRuntime;
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
     * Delete internal structures that require manual cleanup operations;
     */
    public void cleanup() { }
}
