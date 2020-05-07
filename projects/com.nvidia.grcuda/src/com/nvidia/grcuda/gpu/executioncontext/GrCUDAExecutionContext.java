package com.nvidia.grcuda.gpu.executioncontext;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAThreadManager;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyComputationBuilder;
import com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

/**
 * Class used to monitor the state of GrCUDA execution, keep track of memory allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public class GrCUDAExecutionContext extends AbstractGrCUDAExecutionContext {

    /**
     * Reference to the {@link com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager} that takes care of
     * scheduling computations on different streams;
     */
    private final GrCUDAStreamManager streamManager;
    /**
     * Store a reference to the thread manager used to schedule GPU computations;
     */
    private final GrCUDAThreadManager threadManager;

    public GrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, DependencyComputationBuilder dependencyBuilder) {
        this(new CUDARuntime(context, env), new GrCUDAThreadManager(context), dependencyBuilder);
    }

    public GrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAThreadManager threadManager, DependencyComputationBuilder dependencyBuilder) {
        this(cudaRuntime, threadManager, new GrCUDAStreamManager(cudaRuntime), dependencyBuilder);
    }

    public GrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAThreadManager threadManager, GrCUDAStreamManager streamManager, DependencyComputationBuilder dependencyBuilder) {
        super(cudaRuntime, dependencyBuilder);
        this.threadManager = threadManager;
        this.streamManager = streamManager;
    }

    /**
     * Register this computation for future execution by the {@link GrCUDAExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data dependencies;
     */
    @Override
    public Object registerExecution(GrCUDAComputationalElement computation) throws UnsupportedTypeException {
        // Add the new computation to the DAG
        ExecutionDAG.DAGVertex vertex = dag.append(computation);

        // Compute the stream where the computation will be done;
        streamManager.assignStream(vertex);

        // Start the computation;
        return executeComputationSync(vertex);

//        // If the computation can be executed immediately, start it;
//        if (vertex.isExecutable() && threadManager != null) {
//            threadManager.submitRunnable(new ComputationThread(vertex));
//        }
    }

    @Override
    public boolean isAnyComputationActive() {
        return this.streamManager.isAnyComputationActive();
    }

    public GrCUDAStreamManager getStreamManager() {
        return streamManager;
    }

    /**
     * Delete internal structures that require manual cleanup operations;
     */
    @Override
    public void cleanup() {
        threadManager.finalizeManager();
        streamManager.cleanup();
    }

    private Object executeComputationSync(ExecutionDAG.DAGVertex vertex) throws UnsupportedTypeException {
        // Before starting this computation, ensure that all its parents have finished their computation;
        streamManager.syncParentStreams(vertex);

        System.out.println("-- running " + vertex.getComputation());

        // Perform the computation;
        vertex.getComputation().setComputationStarted();
        vertex.getComputation().updateIsComputationArrayAccess();
        return vertex.getComputation().execute();
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
            try {
                vertex.getComputation().execute();
            } catch (UnsupportedTypeException e) {
                e.printStackTrace();
            }
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
