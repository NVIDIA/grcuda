package com.nvidia.grcuda.gpu.executioncontext;

import com.nvidia.grcuda.GPUPointer;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAThreadManager;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.array.DeviceArray;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.computation.KernelExecution;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/**
 * Class used to monitor the state of GrCUDA execution, keep track of memory allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public class MultithreadGrCUDAExecutionContext extends AbstractGrCUDAExecutionContext {

    /**
     * Reference to the {@link GrCUDAStreamManager} that takes care of
     * scheduling computations on different streams;
     */
    private final GrCUDAStreamManager streamManager;
    /**
     * Store a reference to the thread manager used to schedule GPU computations;
     */
    private final GrCUDAThreadManager threadManager;

    // TODO: vertices and threads have a 1:1 mapping, it makes sense to store threads in the vertex.
    //   The DAG must be modified to accept custom vertices, e.g. VertexWithComputationThread;
    protected final HashMap<ExecutionDAG.DAGVertex, CompletableFuture<Object>> activeComputations = new HashMap<>();

    public MultithreadGrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, DependencyPolicyEnum dependencyPolicy) {
        this(new CUDARuntime(context, env), new GrCUDAThreadManager(context), dependencyPolicy);
    }

    public MultithreadGrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAThreadManager threadManager, DependencyPolicyEnum dependencyPolicy) {
        this(cudaRuntime, threadManager, new GrCUDAStreamManager(cudaRuntime), dependencyPolicy);
    }

    public MultithreadGrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAThreadManager threadManager,
                                             GrCUDAStreamManager streamManager, DependencyPolicyEnum dependencyPolicy) {
        super(cudaRuntime, dependencyPolicy);
        this.threadManager = threadManager;
        this.streamManager = streamManager;
    }

    /**
     * Register this computation for future execution by the {@link MultithreadGrCUDAExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data dependencies;
     */
    @Override
    public Object registerExecution(GrCUDAComputationalElement computation) throws UnsupportedTypeException {

        // Add the new computation to the DAG
        ExecutionDAG.DAGVertex vertex = dag.append(computation);

        // Update for each input array if the latest scheduled computation (i.e. this one) is an array access;
        vertex.getComputation().updateIsComputationArrayAccess();

        // Schedule the computation on a thread if it support asynchronous execution on a stream,
        //   else do it synchronously;
        if (vertex.getComputation().canUseStream() && threadManager != null) {

            // Compute the stream where the computation will be done;
            streamManager.assignStream(vertex);

            System.out.println("-- schedule " + vertex.getComputation());
            ComputationThread newThread = getNewComputationThread(vertex);

            if (vertex.isExecutable()) {
                // If the computation can be executed immediately, start it;
                CompletableFuture<Object> futureResult = supplyAsyncComputationThread(newThread);
                activeComputations.put(vertex, futureResult);
            } else {
                // If the computation has dependencies, add the callback logic;
                List<ExecutionDAG.DAGVertex> parents = vertex.getParentVertices();
                CompletableFuture<Void> parentResult = CompletableFuture.allOf(parents.stream().map(activeComputations::get).toArray(CompletableFuture[]::new));
                CompletableFuture<Object> futureResult = parentResult.thenCompose((res) -> supplyAsyncComputationThread(newThread));
                activeComputations.put(vertex, futureResult);
            }
            return NoneValue.get();
        } else {
            return executeComputationSync(vertex);
        }
    }

    protected ComputationThread getNewComputationThread(ExecutionDAG.DAGVertex vertex) {
        return new ComputationThread(vertex);
    }

    private CompletableFuture<Object> supplyAsyncComputationThread(ComputationThread thread) {
        return CompletableFuture.supplyAsync(thread::call, threadManager.getThreadPool())
                .exceptionally(ex -> {
                    System.out.println("Exception encountered in computation thread of " + thread.getVertex().getComputation() + ": " + ex.getMessage());
                    return NoneValue.get();
                });
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

    private Object executeComputationSync(ExecutionDAG.DAGVertex vertex) {
        // Before starting this computation, ensure that all active threads are finished;
        System.out.println("-- running sync " + vertex.getComputation());

        List<ExecutionDAG.DAGVertex> parents = vertex.getParentVertices();
        if (parents.size() == 1) {
            CompletableFuture<Object> result = activeComputations.get(parents.get(0));
            waitForComputationToEnd(result, vertex);
        } else if (parents.size() > 1) {
            System.out.println("\t* must sync " + parents.size() + " parents");
            CompletableFuture<Void> result = CompletableFuture.allOf(parents.stream().map(activeComputations::get).toArray(CompletableFuture[]::new));
            waitForComputationToEnd(result, vertex);
        }
        return executeComputationSyncInternal(vertex);
    }

    private <T> Object waitForComputationToEnd(CompletableFuture<T> result, ExecutionDAG.DAGVertex vertex) {
//        System.out.println("\t* parents iscanc=" + result.isCancelled() + "; isdone=" + result.isDone() + "; isdoneexc=" + result.isCompletedExceptionally());
        try {
            return result.get();
        } catch (InterruptedException | ExecutionException e) {
            System.out.println("Error in sync execution of " + vertex.getComputation() + " = " + e.getMessage());
            e.printStackTrace();
            return NoneValue.get();
        }
    }

    private Object executeComputationSyncInternal(ExecutionDAG.DAGVertex vertex) {
        // Perform the computation;
        vertex.getComputation().setComputationStarted();
        try {
            return vertex.getComputation().execute();
        } catch (UnsupportedTypeException e) {
            System.out.println("Error in sync execution of " + vertex.getComputation() + " = " + e.getMessage());
            e.printStackTrace();
            return NoneValue.get();
        }
    }

    protected class ComputationThread implements Callable<Object> {

        private final ExecutionDAG.DAGVertex vertex;

        public ComputationThread(ExecutionDAG.DAGVertex vertex) {
            this.vertex = vertex;
        }

        protected void setContext() {
            cudaRuntime.cudaSetDevice(0);
        }

        public ExecutionDAG.DAGVertex getVertex() {
            return vertex;
        }

        public Object call() {
            // Perform the computation;
            System.out.println("-- running async " + vertex.getComputation());
            vertex.getComputation().setComputationStarted();
            setContext();
            try {
                vertex.getComputation().execute();
            } catch (UnsupportedTypeException e) {
                System.out.println("Error in async execution of " + vertex.getComputation() + " = " + e.getMessage());
                e.printStackTrace();
                throw new RuntimeException(e);
            }
            // Synchronize on the stream associated to this computation;
            System.out.println("\tsync thread on stream " + vertex.getComputation().getStream() + " by " + vertex.getComputation());
            streamManager.syncStream(vertex.getComputation().getStream());
            vertex.getComputation().setComputationFinished();
            System.out.println("\tfinish sync thread on stream " + vertex.getComputation().getStream());

            return NoneValue.get();
        }
    }
}
