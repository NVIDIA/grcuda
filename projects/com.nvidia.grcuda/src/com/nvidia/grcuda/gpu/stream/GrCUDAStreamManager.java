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
        // If it has a manually specified stream, use it. If the computation cannot use customized streams, return immediately;
        if (!vertex.getComputation().useManuallySpecifiedStream() && vertex.getComputation().canUseStream()) {
            CUDAStream stream;
            if (vertex.isStart()) {
                // Else, if the computation doesn't have parents, provide a new stream to it;
                // TODO: can we do better? E.g. re-use a stream that is not being used
                stream = createStream();
            } else {
                // Else, compute the streams used by the parent computations.
                // If more than one stream is available, keep the first;
                // TODO: this should be more complex!
                //  If 2 child computations have disjoint set of dependencies, they can use 2 streams and run in parallel!
                stream = vertex.getParentComputations().get(0).getStream();
            }
            // Set the stream;
            vertex.getComputation().setStream(stream);
            // Associate all the arrays in the computation to the selected stream,
            //   to enable CPU accesses on managed memory arrays currently not being used by the GPU.
            // This is required as on pre-Pascal GPUs all unified memory pages are locked by the GPU while code is running on the GPU,
            //   even if the GPU is not using some of the pages. Enabling memory-stream association allows the CPU to
            //   access memory not being currently used by a kernel;
            vertex.getComputation().associateArraysToStream();
        }
    }


    /**
     * Given a computation, synchronize all its parent streams. The caller thread will be blocked until all the
     * computations on the parents streams are finished;
     * @param vertex a computation whose parents should be synchronized
     */
    public void syncParentStreams(ExecutionDAG.DAGVertex vertex) {
        vertex.getParentComputations().forEach(c -> {
            // Synchronize computations that are not yet finished and can use streams;
            if (!c.isComputationFinished() && c.canUseStream()) {
//                System.out.println("\tsync thread on stream " + c.getStream() + " by " + vertex.getComputation());
                runtime.cudaStreamSynchronize(c.getStream());
//                System.out.println("\tfinish sync thread on stream " + c.getStream());
                // Set the parent computations as finished;
                c.setComputationFinished();
            }
        });
    }

    /**
     * Create a new {@link CUDAStream} and add it to this manager, then return it;
     */
    public CUDAStream createStream() {
        CUDAStream newStream = runtime.cudaStreamCreate(streams.size());
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
