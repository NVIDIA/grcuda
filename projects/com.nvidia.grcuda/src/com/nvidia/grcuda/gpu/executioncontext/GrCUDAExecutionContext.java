package com.nvidia.grcuda.gpu.executioncontext;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAThreadManager;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.computation.prefetch.PrefetcherEnum;
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

    public GrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, DependencyPolicyEnum dependencyPolicy, PrefetcherEnum inputPrefetch) {
        this(new CUDARuntime(context, env), new GrCUDAThreadManager(context), dependencyPolicy, inputPrefetch);
    }

    public GrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAThreadManager threadManager, DependencyPolicyEnum dependencyPolicy, PrefetcherEnum inputPrefetch) {
        this(cudaRuntime, threadManager, new GrCUDAStreamManager(cudaRuntime), dependencyPolicy, inputPrefetch);
    }

    public GrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAThreadManager threadManager, GrCUDAStreamManager streamManager, DependencyPolicyEnum dependencyPolicy) {
        super(cudaRuntime, dependencyPolicy, PrefetcherEnum.NONE, ExecutionPolicyEnum.DEFAULT);
        this.streamManager = streamManager;
    }

    public GrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAThreadManager threadManager, GrCUDAStreamManager streamManager, DependencyPolicyEnum dependencyPolicy, PrefetcherEnum inputPrefetch) {
        super(cudaRuntime, dependencyPolicy, inputPrefetch, ExecutionPolicyEnum.DEFAULT);
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

        // Compute the stream where the computation will be done, if the computation can be performed asynchronously;
        streamManager.assignStream(vertex);

        // Prefetching;
        arrayPrefetcher.prefetchToGpu(vertex);

        // Start the computation;
        Object result = executeComputationSync(vertex);

        // Associate a CUDA event to this computation, if performed asynchronously;
        streamManager.assignEvent(vertex);

//        System.out.println("-- running " + vertex.getComputation());

        return result;
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
        streamManager.cleanup();
    }

    private Object executeComputationSync(ExecutionDAG.DAGVertex vertex) throws UnsupportedTypeException {
        // Before starting this computation, ensure that all its parents have finished their computation;
        streamManager.syncParentStreams(vertex);

        // Perform the computation;
        vertex.getComputation().setComputationStarted();
        vertex.getComputation().updateIsComputationArrayAccess();
        return vertex.getComputation().execute();
    }
}
