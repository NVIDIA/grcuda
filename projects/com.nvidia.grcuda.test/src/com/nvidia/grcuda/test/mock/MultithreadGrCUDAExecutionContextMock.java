package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.computation.ArrayStreamArchitecturePolicy;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.computation.PrePascalArrayStreamAssociation;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.gpu.executioncontext.MultithreadGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/**
 * Mock class to test the GrCUDAExecutionContextTest, it has a null CUDARuntime;
 */
public class MultithreadGrCUDAExecutionContextMock extends MultithreadGrCUDAExecutionContext {

    private static final int THREAD_NUM = 16;

    public MultithreadGrCUDAExecutionContextMock(boolean syncStream) {
        super(null, new GrCUDAThreadManagerMock(THREAD_NUM),
                new GrCUDAStreamManagerMock(null, syncStream), DependencyPolicyEnum.DEFAULT);
    }

    public MultithreadGrCUDAExecutionContextMock() {
        super(null, new GrCUDAThreadManagerMock(THREAD_NUM),
                new GrCUDAStreamManagerMock(null), DependencyPolicyEnum.DEFAULT);
    }

    public MultithreadGrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy) {
        super(null, new GrCUDAThreadManagerMock(THREAD_NUM),
                new GrCUDAStreamManagerMock(null), dependencyPolicy);
    }

    public MultithreadGrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy, boolean syncStream,
                                                 RetrieveNewStreamPolicyEnum retrieveStreamPolicy,
                                                 RetrieveParentStreamPolicyEnum parentStreamPolicyEnum) {
        super(null, new GrCUDAThreadManagerMock(THREAD_NUM),
                new GrCUDAStreamManagerMock(null, syncStream, retrieveStreamPolicy, parentStreamPolicyEnum), dependencyPolicy);
    }

    public ArrayStreamArchitecturePolicy getArrayStreamArchitecturePolicy() {
        return new PrePascalArrayStreamAssociation();
    }

    // Used to wait the end of a specific computation;
    public void waitFinish(GrCUDAComputationalElement computation) {
        System.out.println("-- forcing wait of " + computation);
        for (Map.Entry<ExecutionDAG.DAGVertex, CompletableFuture<Object>> entry : activeComputations.entrySet()) {
            if (entry.getKey().getComputation().equals(computation)) {
                try {
                    CompletableFuture<Object> task = entry.getValue();
                    task.get();
                } catch (InterruptedException | ExecutionException e) {
                    System.out.println("failed to wait for computation " + computation + " to finish");
                    e.printStackTrace();
                }
            }
        }
    }

    @Override
    protected ComputationThread getNewComputationThread(ExecutionDAG.DAGVertex vertex) {
        return new ComputationThreadMock(vertex);
    }

    protected class ComputationThreadMock extends ComputationThread {

        public ComputationThreadMock(ExecutionDAG.DAGVertex vertex) {
            super(vertex);
        }

        // Don't do anything;
        protected void setContext() { }
    }
}
