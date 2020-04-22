package com.nvidia.grcuda.gpu.stream;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.ExecutionDAG;

import java.util.ArrayList;
import java.util.List;

public class GrCUDAStreamManager {

    /**
     * List of {@link CUDAStream} that have been currently allocated;
     */
    protected List<CUDAStream> streams = new ArrayList<>();
    /**
     * Reference to the CUDA runtime that manages the streams;
     */
    protected CUDARuntime runtime;

    public GrCUDAStreamManager(CUDARuntime runtime) {
        this.runtime = runtime;
    }

    /**
     * Assign a {@link CUDAStream} to the input computation, based on its dependencies and on the available streams.
     * This function has no effect if the stream was manually specified by the user;
     * @param vertex an input computation for which we want to assign a stream
     */
    public void assignStream(ExecutionDAG.DAGVertex vertex) {
        // If it has a manually specified stream, use it;
        if (!vertex.getComputation().useManuallySpecifiedStream()) {
            if (vertex.isStart()) {
                // Else, if the computation doesn't have parents, provide a new stream to it;
                // TODO: can we do better? E.g. re-use a stream that is not being used
                vertex.getComputation().setStream(createStream());
            } else {
                // Else, compute the streams used by the parent computations.
                // If more than one stream is available, keep the first;
                // TODO: this should be more complex!
                //  If 2 child computations have disjoint set of dependencies, they can use 2 streams and run in parallel!
                CUDAStream stream = vertex.getParentComputations().get(0).getStream();
                vertex.getComputation().setStream(stream);
            }
        }
    }

    /**
     * Given a computation, synchronize all its parent streams. The caller thread will be blocked until all the
     * computations on the parents streams are finished;
     * @param vertex a computation whose parents should be synchronized
     */
    public void syncParentStreams(ExecutionDAG.DAGVertex vertex) {
        vertex.getParentComputations().forEach(c -> {
            if (!c.isComputationFinished()) {
                System.out.println("\tsync thread on stream " + c.getStream());
                runtime.cudaStreamSynchronize(c.getStream());
                System.out.println("\tfinish sync thread on stream " + c.getStream());
                // Set the parent computations to be finished;
                c.setComputationFinished();
            }
        });
    }

    /**
     * Create a new {@link CUDAStream} and add it to this manager, then return it;
     */
    public CUDAStream createStream() {
        CUDAStream newStream = runtime.cudaStreamCreate();
        streams.add(newStream);
        return newStream;
    }

    /**
     * Obtain the number of streams managed by this manager;
     */
    public int getNumberOfStreams() {
        return streams.size();
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        streams.forEach(runtime::cudaStreamDestroy);
    }
}
