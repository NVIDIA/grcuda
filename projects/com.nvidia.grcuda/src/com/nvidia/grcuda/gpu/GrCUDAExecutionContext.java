package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAThreadManager;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager;
import com.oracle.truffle.api.TruffleLanguage;

import java.util.HashSet;
import java.util.Set;

/**
 * Class used to monitor the state of GrCUDA execution, keep track of memory allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public class GrCUDAExecutionContext {

    /**
     * Reference to the inner {@link CUDARuntime} used to execute kernels and other {@link GrCUDAComputationalElement}
     */
    private final CUDARuntime cudaRuntime;
    /**
     * Reference to the {@link com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager} that takes care of
     * scheduling computations on different streams;
     */
    private final GrCUDAStreamManager streamManager;
    /**
     * Store a reference to the thread manager used to schedule GPU computations;
     */
    private final GrCUDAThreadManager threadManager;

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

    public GrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env) {
        this(new CUDARuntime(context, env), new GrCUDAThreadManager(context));
    }

    public GrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAThreadManager threadManager) {
        this(cudaRuntime, threadManager, new GrCUDAStreamManager(cudaRuntime));
    }

    public GrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAThreadManager threadManager, GrCUDAStreamManager streamManager) {
        this.cudaRuntime = cudaRuntime;
        this.threadManager = threadManager;
        this.streamManager = streamManager;
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
    public void registerExecution(GrCUDAComputationalElement computation) {
        // Add the new computation to the DAG
        ExecutionDAG.DAGVertex vertex = dag.append(computation);

        // Compute the stream where the computation will be done;
        streamManager.assignStream(vertex);

        // Start the computation;
        executeComputationSync(vertex);
//
//        // If the computation can be executed immediately, start it;
//        if (vertex.isExecutable() && threadManager != null) {
//            threadManager.submitRunnable(new ComputationThread(vertex));
//        }
    }

    public ExecutionDAG getDag() {
        return dag;
    }

    public CUDARuntime getCudaRuntime() {
        return cudaRuntime;
    }

    public GrCUDAStreamManager getStreamManager() {
        return streamManager;
    }

    // Functions used to interface directly with the CUDA runtime;

    public Kernel loadKernel(String cubinFile, String kernelName, String signature) {
        return cudaRuntime.loadKernel(this, cubinFile, kernelName, signature);
    }

    public Kernel buildKernel(String code, String kernelName, String signature) {
        return cudaRuntime.buildKernel(this, code, kernelName, signature);
    }

    /**
     * Delete internal structures that require manual cleanup operations;
     */
    public void cleanup() {
        threadManager.finalizeManager();
        streamManager.cleanup();
    }

    private void executeComputationSync(ExecutionDAG.DAGVertex vertex) {
        // Before starting this computation, ensure that all its parents have finished their computation;
        streamManager.syncParentStreams(vertex);

        // Perform the computation;
        System.out.println("Starting execution of " + vertex.getComputation());
        vertex.getComputation().setComputationStarted();
        vertex.getComputation().execute();
    }

    private class ComputationThread implements Runnable {

        private final ExecutionDAG.DAGVertex vertex;

        public ComputationThread(ExecutionDAG.DAGVertex vertex) {
            this.vertex = vertex;
        }

        public void run(){
            // Perform the computation;
            System.out.println("Starting execution of " + vertex.getComputation());
            vertex.getComputation().setComputationStarted();
            vertex.getComputation().execute();
            // Synchronize on the stream associated to this computation;
            System.out.println("\tsync thread on stream " + vertex.getComputation().getStream());
            cudaRuntime.cudaStreamSynchronize(vertex.getComputation().getStream());
            vertex.getComputation().setComputationFinished();
            System.out.println("\tfinish sync thread on stream " + vertex.getComputation().getStream());
            // Update the status of this computation and of its children;
            vertex.getChildVertices().forEach(v -> {
                if (v.isExecutable() && threadManager != null) {
                    threadManager.submitRunnable(new ComputationThread(v));
                }
            });
        }
    }
}
