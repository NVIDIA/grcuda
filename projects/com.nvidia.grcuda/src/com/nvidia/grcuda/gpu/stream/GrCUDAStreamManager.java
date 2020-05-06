package com.nvidia.grcuda.gpu.stream;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.Set;

public class GrCUDAStreamManager {

    /**
     * List of {@link CUDAStream} that have been currently allocated;
     */
    protected List<CUDAStream> streams = new ArrayList<>();
    /**
     * Reference to the CUDA runtime that manages the streams;
     */
    protected final CUDARuntime runtime;
    /**
     * Track the active computations each stream has, excluding the default stream;
     */
    protected final Map<CUDAStream, Set<GrCUDAComputationalElement>> activeComputationsPerStream = new HashMap<>();

    private final RetrieveStream retrieveStream;

    // TODO: tests on the 2 retrieve stream policies, with extra tests on LIFO;
    public GrCUDAStreamManager(CUDARuntime runtime) {
        this(runtime, runtime.getContext().getRetrieveStreamPolicy());
    }

    public GrCUDAStreamManager(CUDARuntime runtime, RetrieveStreamPolicyEnum retrieveStreamPolicyEnum) {
        this.runtime = runtime;
        switch(retrieveStreamPolicyEnum) {
            case LIFO:
                this.retrieveStream = new LifoRetrieveStream();
                break;
            case ALWAYS_NEW:
                this.retrieveStream = new AlwaysNewRetrieveStream();
                break;
            default:
                this.retrieveStream = new LifoRetrieveStream();
        }
    }

    /**
     * Assign a {@link CUDAStream} to the input computation, based on its dependencies and on the available streams.
     * This function has no effect if the stream was manually specified by the user;
     * @param vertex an input computation for which we want to assign a stream
     */
    public void assignStream(ExecutionDAG.DAGVertex vertex) {

        // If the computation cannot use customized streams, return immediately;
        if (vertex.getComputation().canUseStream()) {
            CUDAStream stream;
            if (vertex.isStart()) {
                // Else, if the computation doesn't have parents, provide a new stream to it;
                stream = retrieveStream.retrieve();
            } else {
                // Else, compute the streams used by the parent computations.
                // If more than one stream is available, keep the first;
                // TODO: this should be more complex!
                //  If 2 child computations have disjoint set of dependencies, they can use 2 streams and run in parallel!
                stream = vertex.getParentComputations().get(0).getStream();
            }
            // Set the stream;
            vertex.getComputation().setStream(stream);
            // Update the computation counter;
            addActiveComputation(vertex.getComputation());
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
        // FIXME: if I write on array x, launch K(Y) then read(x), the last comp on x is array access, so no sync is done!!!

        // Skip syncing if no computation is active;
        // FIXME: this doesn't check if streams are empty! it works only at the very start
        if (!activeComputationsPerStream.isEmpty()) {
            Set<GrCUDAComputationalElement> computationsToSync = new HashSet<>(vertex.getParentComputations());

            // Retrieve an additional stream dependency from the kernel, if required;
            Optional<CUDAStream> additionalStream = vertex.getComputation().additionalStreamDependency();
            if (additionalStream.isPresent()) {
                // If we require synchronization on the default stream, perform it in a specialized way;
                if (additionalStream.get().isDefaultStream()) {
                    System.out.println("--\tsync stream " + additionalStream.get() + " by " + vertex.getComputation());
                    // Synchronize the device;
                    runtime.cudaDeviceSynchronize();
                    // All computations are now finished;
                    resetActiveComputationState();
                } else {
                    // Else add the computations related to the additional streams to the set and sync it;
                    System.out.println("--\tsyncing additional stream " + additionalStream.get() + "...");
                    computationsToSync.addAll(activeComputationsPerStream.get(additionalStream.get()));
                    syncParentStreamsImpl(computationsToSync, vertex.getComputation());
                }
            } else {
                syncParentStreamsImpl(computationsToSync, vertex.getComputation());
            }
        }
    }

    /**
     * Synchronize a list of computations on their streams;
     * @param computationsToSync a list of computations whose streams should be synced
     * @param computationThatSyncs the computation that triggered the syncing process
     */
    private void syncParentStreamsImpl(
            Set<GrCUDAComputationalElement> computationsToSync,
            GrCUDAComputationalElement computationThatSyncs) {
        computationsToSync.forEach(c -> {
            // Synchronize computations that are not yet finished and can use streams;
            if (c.canUseStream() && !c.isComputationFinished()) {
                System.out.println("--\tsync stream " + c.getStream() + " by " + computationThatSyncs);
                runtime.cudaStreamSynchronize(c.getStream());
                // Set the parent computations as finished;
                c.setComputationFinished();
                // Decrement the active computation count;
                removeActiveComputation(c);
            }
        });
    }

    /**
     * Create a new {@link CUDAStream} and add it to this manager, then return it;
     */
    public CUDAStream createStream() {
        CUDAStream newStream = runtime.cudaStreamCreate(streams.size());
        streams.add(newStream);
//        activeComputationsPerStream.put(newStream, new HashSet<>());
        return newStream;
    }

    /**
     * Obtain the number of streams managed by this manager;
     */
    public int getNumberOfStreams() {
        return streams.size();
    }

    public int getNumActiveComputationsOnStream(CUDAStream stream) {
        return activeComputationsPerStream.get(stream).size();
    }

    protected void addActiveComputation(GrCUDAComputationalElement computation) {
        CUDAStream stream = computation.getStream();
        // Start tracking the stream if it wasn't already tracked;
        if (!activeComputationsPerStream.containsKey(stream)) {
            activeComputationsPerStream.put(stream, new HashSet<>());
        }
        // Associate the computation to the stream;
        activeComputationsPerStream.get(stream).add(computation);
    }

    /**
     * Remove a computation from the map that associates streams to their active computations,
     * and mark the stream as free if no other computations are active on the stream;
     * @param computation a computation that is no longer active
     */
    protected void removeActiveComputation(GrCUDAComputationalElement computation) {
        CUDAStream stream = computation.getStream();
        activeComputationsPerStream.get(stream).remove(computation);
        // If this stream doesn't have any computation associated to it, it's free to use;
        if (activeComputationsPerStream.get(stream).isEmpty()) {
            activeComputationsPerStream.remove(stream);
            retrieveStream.update(stream);
        }
    }

    /**
     * Reset the association between streams and computations. All computations are finished, and all streams are free;
     */
    private void resetActiveComputationState() {
        activeComputationsPerStream.keySet().forEach(s -> {
            activeComputationsPerStream.get(s).forEach(GrCUDAComputationalElement::setComputationFinished);
//            activeComputationsPerStream.put(s, new HashSet<>());
        });
        // Streams don't have any active computation;
        activeComputationsPerStream.clear();
        // All streams are free;
        retrieveStream.update(streams);
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        streams.forEach(runtime::cudaStreamDestroy);
        activeComputationsPerStream.clear();
        retrieveStream.cleanup();
        streams.clear();
    }

    /**
     * By default, create a new stream every time;
     */
    private class AlwaysNewRetrieveStream extends RetrieveStream {

        @Override
        public CUDAStream retrieve() {
            return createStream();
        }
    }

    /**
     * Keep a queue of free (currently not utilized) streams, and retrieve the oldest one added to the queue;
     */
    private class LifoRetrieveStream extends RetrieveStream {

        /**
         * Keep a queue of free streams;
         */
        private final Queue<CUDAStream> freeStreams = new ArrayDeque<>();

        @Override
        void update(CUDAStream stream) {
            freeStreams.add(stream);
        }

        @Override
        void update(Collection<CUDAStream> streams) {
            freeStreams.addAll(streams);
        }

        @Override
        CUDAStream retrieve() {
            if (freeStreams.isEmpty()) {
                // Create a new stream if none is available;
                return createStream();
            } else {
                // Get the first stream available, and remove it from the list of free streams;
                return freeStreams.poll();
            }
        }
    }
}
