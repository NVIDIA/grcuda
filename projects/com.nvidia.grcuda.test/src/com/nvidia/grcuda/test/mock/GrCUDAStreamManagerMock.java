package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.CUDAEvent;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class GrCUDAStreamManagerMock extends GrCUDAStreamManager {

    GrCUDAStreamManagerMock(CUDARuntime runtime, boolean syncParents,
                            RetrieveNewStreamPolicyEnum retrieveStreamPolicy,
                            RetrieveParentStreamPolicyEnum parentStreamPolicyEnum) {
        super(runtime, retrieveStreamPolicy, parentStreamPolicyEnum);
        this.syncParents = syncParents;
    }

    GrCUDAStreamManagerMock(CUDARuntime runtime, boolean syncParents,
                            RetrieveNewStreamPolicyEnum retrieveStreamPolicy) {
        super(runtime, retrieveStreamPolicy, RetrieveParentStreamPolicyEnum.DEFAULT);
        this.syncParents = syncParents;
    }

    GrCUDAStreamManagerMock(CUDARuntime runtime, boolean syncParents) {
        super(runtime, RetrieveNewStreamPolicyEnum.ALWAYS_NEW, RetrieveParentStreamPolicyEnum.DEFAULT);
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
    public void syncStream(CUDAStream stream) { }

    @Override
    protected void syncDevice() { }

    @Override
    protected void syncStreamsUsingEvents(ExecutionDAG.DAGVertex vertex) {
        for (GrCUDAComputationalElement parent : vertex.getParentComputations()) {
            CUDAStream stream = parent.getStream();
            // Skip synchronization on the same stream where the new computation is executed,
            //   as operations scheduled on a stream are executed in order;
            if (!vertex.getComputation().getStream().equals(stream)) {
                // Track that the current computation has a parent executed on a different stream;
                trackAdditionalComputationsToFinish(vertex.getComputation(), parent);
            }
        }
    }

    public Map<CUDAStream, Set<GrCUDAComputationalElement>> getActiveComputationsMap() {
        return this.activeComputationsPerStream;
    }
}
