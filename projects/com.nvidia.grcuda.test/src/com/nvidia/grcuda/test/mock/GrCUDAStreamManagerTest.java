package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager;

import java.util.HashSet;

public class GrCUDAStreamManagerTest extends GrCUDAStreamManager {
    GrCUDAStreamManagerTest(CUDARuntime runtime, boolean syncParents) {
        super(runtime);
        this.syncParents = syncParents;
    }

    GrCUDAStreamManagerTest(CUDARuntime runtime) {
        this(runtime, false);
    }

    int numStreams = 0;

    final boolean syncParents;

    @Override
    public CUDAStream createStream() {
        CUDAStream newStream = new CUDAStream(0, numStreams++);
        streams.add(newStream);
        this.activeComputationsPerStream.put(newStream, new HashSet<>());
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
}
