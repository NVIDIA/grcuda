package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class GrCUDAStreamManagerMock extends GrCUDAStreamManager {

    GrCUDAStreamManagerMock(CUDARuntime runtime,
                            RetrieveNewStreamPolicyEnum retrieveStreamPolicy,
                            RetrieveParentStreamPolicyEnum parentStreamPolicyEnum) {
        super(runtime, retrieveStreamPolicy, parentStreamPolicyEnum);
    }

    GrCUDAStreamManagerMock(CUDARuntime runtime,
                            RetrieveNewStreamPolicyEnum retrieveStreamPolicy) {
        super(runtime, retrieveStreamPolicy, RetrieveParentStreamPolicyEnum.SAME_AS_PARENT);
    }

    GrCUDAStreamManagerMock(CUDARuntime runtime) {
        super(runtime, RetrieveNewStreamPolicyEnum.ALWAYS_NEW, RetrieveParentStreamPolicyEnum.SAME_AS_PARENT);
    }

    int numStreams = 0;

    @Override
    public CUDAStream createStream() {
        CUDAStream newStream = new CUDAStream(0, numStreams++);
        streams.add(newStream);
        return newStream;
    }

    @Override
    public void assignEvent(ExecutionDAG.DAGVertex vertex) { }

    @Override
    public void syncStream(CUDAStream stream) { }

    @Override
    protected void setComputationFinishedInner(GrCUDAComputationalElement computation) {
        computation.setComputationFinished();
    }

    @Override
    protected void syncStreamsUsingEvents(ExecutionDAG.DAGVertex vertex) { }

    @Override
    protected void syncDevice() { }

    public List<CUDAStream> getStreams() { return this.streams; }

    public Map<CUDAStream, Set<GrCUDAComputationalElement>> getActiveComputationsMap() {
        Map<CUDAStream, Set<GrCUDAComputationalElement>> activeComputations = new HashMap<>();
        for (Map.Entry<CUDAStream, Set<ExecutionDAG.DAGVertex>> e : this.activeComputationsPerStream.entrySet()) {
            activeComputations.put(e.getKey(), e.getValue().stream().map(ExecutionDAG.DAGVertex::getComputation).collect(Collectors.toSet()));
        }
        return activeComputations;
    }
}
