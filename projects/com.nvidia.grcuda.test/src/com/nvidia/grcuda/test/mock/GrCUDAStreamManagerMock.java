package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager;
import com.nvidia.grcuda.gpu.stream.RetrieveStreamPolicyEnum;

import java.util.Map;
import java.util.Set;

public class GrCUDAStreamManagerMock extends GrCUDAStreamManager {

    GrCUDAStreamManagerMock(CUDARuntime runtime, boolean syncParents, RetrieveStreamPolicyEnum retrieveStreamPolicy) {
        super(runtime, retrieveStreamPolicy);
        this.syncParents = syncParents;
    }

    GrCUDAStreamManagerMock(CUDARuntime runtime, boolean syncParents) {
        super(runtime, RetrieveStreamPolicyEnum.ALWAYS_NEW);
        this.syncParents = syncParents;
    }

    GrCUDAStreamManagerMock(CUDARuntime runtime) {
        this(runtime, false);
    }

    int numStreams = 0;

    final boolean syncParents;

    @Override
    public CUDAStream createStream() {
        CUDAStream newStream = new CUDAStream(0, numStreams++);
        streams.add(newStream);
        return newStream;
    }

    @Override
    public void syncParentStreams(ExecutionDAG.DAGVertex vertex) {
        if (syncParents) {
            vertex.getParentComputations().forEach(c -> {
                // Synchronize computations that are not yet finished and can use streams;
                if (!c.isComputationFinished() && c.canUseStream()) {
                    // Set the parent computations as finished;
                    c.setComputationFinished();
                    // Decrement the active computation count;
                    removeActiveComputation(c);
                }
            });
        }
    }

    public Map<CUDAStream, Set<GrCUDAComputationalElement>> getActiveComputationsMap() {
        return this.activeComputationsPerStream;
    }
}
