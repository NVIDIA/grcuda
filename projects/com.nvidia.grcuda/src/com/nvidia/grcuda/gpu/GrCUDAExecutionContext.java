package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAThreadManager;
import com.nvidia.grcuda.array.AbstractArray;
import com.oracle.truffle.api.TruffleLanguage;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.Callable;

/**
 * Class used to monitor the state of GrCUDA execution, keep track of memory allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public class GrCUDAExecutionContext {

    /**
     * Reference to the inner {@link CUDARuntime} used to execute kernels and other {@link GrCUDAComputationalElement}
     */
    private final CUDARuntime cudaRuntime;

    private GrCUDAThreadManager threadManager;

    /**
     * Set that contains all the arrays allocated so far.
     */
    final private Set<AbstractArray> arraySet = new HashSet<>();

    /**
     * Set that contains all the CUDA kernels declared so far.
     */
    final private Set<Kernel> kernelSet = new HashSet<>();

    /**
     * Set that contains all the CUDA kernels execution so far.
     * TODO: this should not be a set, but a DAG that can be used to handle dependencies
     */
    final private Set<GrCUDAComputationalElement> kernelExecSet = new HashSet<>();

    final private ExecutionDAG dag = new ExecutionDAG();

    public GrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, GrCUDAThreadManager threadManager) {
        this.cudaRuntime = new CUDARuntime(context, env);
        this.threadManager = threadManager;
    }

    public GrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAThreadManager threadManager) {
        this.cudaRuntime = cudaRuntime;
        this.threadManager = threadManager;
    }

    public void registerArray(AbstractArray array) {
        arraySet.add(array);
    }

    public void registerKernel(Kernel kernel) {
        kernelSet.add(kernel);
    }

    /**
     * Register this computation for future execution by the {@link GrCUDAExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data dependencies;
     */
    public void registerExecution(GrCUDAComputationalElement kernel) {
        dag.append(kernel);

        if (threadManager != null) {
            cudaRuntime.cudaDeviceSynchronize();
            kernel.execute();
            cudaRuntime.cudaDeviceSynchronize();
        }
//        // TODO: here should go the scheduling logic, and decide if/when the thread is spawned and under which conditions;
//        if (threadManager != null) {
//            threadManager.submitRunnable(() -> {
//                cudaRuntime.cudaDeviceSynchronize();
//                kernel.execute();
////                cudaRuntime.cudaDeviceSynchronize();
//            });
//        }
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
}
