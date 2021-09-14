package com.nvidia.grcuda.gpu.stream;

import com.nvidia.grcuda.CUDAEvent;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

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
    protected final Map<CUDAStream, Set<ExecutionDAG.DAGVertex>> activeComputationsPerStream = new HashMap<>();

    private final RetrieveNewStream retrieveNewStream;
    private final RetrieveParentStream retrieveParentStream;

    public GrCUDAStreamManager(CUDARuntime runtime) { 
        this(runtime, runtime.getContext().getRetrieveNewStreamPolicy(), runtime.getContext().getRetrieveParentStreamPolicyEnum());
    }

    public GrCUDAStreamManager(
            CUDARuntime runtime,
            RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
            RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum) {
        this.runtime = runtime;
        // Get how streams are retrieved for computations without parents;
        switch(retrieveNewStreamPolicyEnum) {
            case FIFO:
                this.retrieveNewStream = new FifoRetrieveStream();
                break;
            case ALWAYS_NEW:
                this.retrieveNewStream = new AlwaysNewRetrieveStream();
                break;
            default:
                this.retrieveNewStream = new FifoRetrieveStream();
        }
        // Get how streams are retrieved for computations with parents;
        switch(retrieveParentStreamPolicyEnum) {
            case DISJOINT:
                this.retrieveParentStream = new DisjointRetrieveParentStream(this.retrieveNewStream);
                break;
            case SAME_AS_PARENT:
                this.retrieveParentStream = new DefaultRetrieveParentStream();
                break;
            default:
                this.retrieveParentStream = new DefaultRetrieveParentStream();
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
                stream = this.retrieveNewStream.retrieve();
            } else {
                // Else, compute the streams used by the parent computations.
                stream = this.retrieveParentStream.retrieve(vertex);
            }
            // Set the stream;
            vertex.getComputation().setStream(stream);
            // Update the computation counter;
            addActiveComputation(vertex);
            // Associate all the arrays in the computation to the selected stream,
            //   to enable CPU accesses on managed memory arrays currently not being used by the GPU.
            // This is required as on pre-Pascal GPUs all unified memory pages are locked by the GPU while code is running on the GPU,
            //   even if the GPU is not using some of the pages. Enabling memory-stream association allows the CPU to
            //   access memory not being currently used by a kernel;
            vertex.getComputation().associateArraysToStream();
        }
    }

    /**
     * Associate a new {@link CUDAEvent} to this computation, if the computation is done on a {@link CUDAStream}.
     * The event is created and recorded on the stream where the computation is running,
     * and can be used for precise synchronization of children computation;
     * @param vertex an input computation for which we want to assign an event
     */
    public void assignEvent(ExecutionDAG.DAGVertex vertex) {
        // If the computation cannot use customized streams, return immediately;
        if (vertex.getComputation().canUseStream()) {
            CUDAEvent event = runtime.cudaEventCreate();
            runtime.cudaEventRecord(event, vertex.getComputation().getStream());
            vertex.getComputation().setEvent(event);
        }
    }

    public void syncParentStreams(ExecutionDAG.DAGVertex vertex) {
        // If the vertex can be executed on a CUDA stream, use CUDA events,
        //   otherwise use stream/device synchronization to block the host until synchronization is done;
        if (vertex.getComputation().canUseStream()) {
            syncStreamsUsingEvents(vertex);
        } else {
            if (this.isAnyComputationActive()) {
                Optional<CUDAStream> additionalStream = vertex.getComputation().additionalStreamDependency();
                if (additionalStream.isPresent()) {
                    CUDAStream stream = additionalStream.get();
                    // If we require synchronization on the default stream, perform it in a specialized way;
                    if (stream.isDefaultStream()) {
//                        System.out.println("--\tsync stream " + stream + " by " + vertex.getComputation());
                        // Synchronize the device;
                        syncDevice();
                        // All computations are now finished;
                        resetActiveComputationState();
                    } else {
                        // Else add the computations related to the additional streams to the set and sync it,
                        //   as long as the additional stream isn't the same as the one that we have to sync anyway;
                        syncParentStreamsImpl(vertex);
                    }
                } else {
                    syncParentStreamsImpl(vertex);
                }
            }
        }
    }

    /**
     * Obtain the set of CUDAStreams that have to be synchronized;
     * @param computationsToSync a set of computations to sync
     * @return the set of CUDAStreams that have to be synchronized
     */
    protected Set<CUDAStream> getParentStreams(Collection<GrCUDAComputationalElement> computationsToSync) {
        return computationsToSync.stream().map(GrCUDAComputationalElement::getStream).collect(Collectors.toSet());
    }

    /**
     * If a computation can be scheduled on a stream, use {@link CUDAEvent} to synchronize parent computations
     * without blocking the CPU host
     * @param vertex a computation whose parent's streams must be synchronized
     */
    protected void syncStreamsUsingEvents(ExecutionDAG.DAGVertex vertex) {
        for (GrCUDAComputationalElement parent : vertex.getParentComputations()) {
            CUDAStream stream = parent.getStream();
            // Skip synchronization on the same stream where the new computation is executed,
            //   as operations scheduled on a stream are executed in order;
            if (!vertex.getComputation().getStream().equals(stream)) {
                // Synchronize on the events associated to the parents;
                if (parent.getEvent().isPresent()) {
                    CUDAEvent event = parent.getEvent().get();
                    runtime.cudaStreamWaitEvent(vertex.getComputation().getStream(), event);

//                    System.out.println("\t* wait event on stream; stream to sync=" + stream.getStreamNumber()
//                            + "; stream that waits=" + vertex.getComputation().getStream().getStreamNumber()
//                            + "; event=" + event.getEventNumber());
                } else {
                    System.out.println("\t* WARNING: missing event to sync child computation=" + vertex.getComputation() +
                            " and parent computation=" + parent);
                }
            }
        }
    }

    /**
     * Synchronization is done in 2 parts:
     * 1. Synchronize the streams where each parent computation is executed;
     * 2. All computations currently active on the synchronized streams are finished, and so are their parents.
     *   In this phase, check if any parent is executed on a stream different from the ones we synchronized, and store these streams.
     *   Also set the streams that have no active computation as free
     * @param vertex the vertex whose parents should be synchronized
     */
    protected void syncParentStreamsImpl(ExecutionDAG.DAGVertex vertex) {

        Set<CUDAStream> streamsToSync = getParentStreams(vertex.getParentComputations());
        // Synchronize streams;
        streamsToSync.forEach(s -> {
//            System.out.println("--\tsync stream=" + s.getStreamNumber() + " by " + vertex.getComputation());
            syncStream(s);
        });

        // Book-keeping: all computations on the synchronized streams are guaranteed to be finished;
        streamsToSync.forEach(s -> {
            activeComputationsPerStream.get(s).forEach(v -> {
                // Skip computations that have already finished;
                if (!v.getComputation().isComputationFinished()) {
                    setComputationsFinished(v, streamsToSync);
                }
            });
            // Now the stream is free to be re-used;
            activeComputationsPerStream.remove(s);
            retrieveNewStream.update(s);
        });
    }

    protected void setComputationFinishedInner(GrCUDAComputationalElement computation) {
        computation.setComputationFinished();
        // Destroy the event associated to this computation;
        if (computation.getEvent().isPresent()) {
            runtime.cudaEventDestroy(computation.getEvent().get());
        } else {
            System.out.println("\t* WARNING: missing event to destroy for computation=" + computation);
        }
    }

    private void setComputationsFinished(ExecutionDAG.DAGVertex vertex, Set<CUDAStream> streamsToSync) {
        // Vertices to process;
        final Queue<ExecutionDAG.DAGVertex> queue = new ArrayDeque<>(Collections.singletonList(vertex));
        // Vertices that have already been seen;
        final Set<ExecutionDAG.DAGVertex> seen = new HashSet<>(Collections.singletonList(vertex));
        // Perform a reverse BFS to process all the parents of the starting computation;
        while (!queue.isEmpty()) {
            ExecutionDAG.DAGVertex currentVertex = queue.poll();
            setComputationFinishedInner(currentVertex.getComputation());
            // Book-keeping on the stream of the current computation;
            CUDAStream stream = currentVertex.getComputation().getStream();

            // Skip streams that have already been synchronized, as they will be freed later;
            if (!streamsToSync.contains(stream)) {
                // Stop considering this computation as active on its stream;
                activeComputationsPerStream.get(stream).remove(currentVertex);
                // If this stream doesn't have any computation associated to it, it's free to use;
                if (activeComputationsPerStream.get(stream).isEmpty()) {
                    activeComputationsPerStream.remove(stream);
                    retrieveNewStream.update(stream);
                }
            }

            // Process parents of the current computation;
            for (ExecutionDAG.DAGVertex parent : currentVertex.getParentVertices()) {
                if (!parent.getComputation().isComputationFinished() && !seen.contains(parent)) {
                    queue.add(parent);
                    seen.add(parent);
                }
            }
        }
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
     * Check if a given stream is free to use, and has no active computations on it;
     * @param stream a CUDAStream
     * @return if the stream has no active computations on it
     */
    public boolean isStreamFree(CUDAStream stream) throws IllegalStateException {
        if (activeComputationsPerStream.containsKey(stream)) {
            if (activeComputationsPerStream.get(stream).isEmpty()) {
                // The stream cannot be in the map without at least one active computation;
                throw new IllegalStateException("stream " + stream.getStreamNumber() + " is tracked but has 0 active computations");
            } else {
                return false; // Stream is active;
            }
        } else {
            return true; // Stream is not active;
        }
    }

    public void syncStream(CUDAStream stream) {
        runtime.cudaStreamSynchronize(stream);
    }

    protected void syncDevice() {
        runtime.cudaDeviceSynchronize();
    }

    /**
     * Obtain the number of streams managed by this manager;
     */
    public int getNumberOfStreams() {
        return streams.size();
    }

    public int getNumActiveComputationsOnStream(CUDAStream stream) {
        if (this.isStreamFree(stream)) {
            return 0;
        } else {
            return activeComputationsPerStream.get(stream).size();
        }
    }

    /**
     * Check if any computation is currently marked as active, and is running on a stream.
     * If so, scheduling of new computations is likely to require synchronizations of some sort;
     * @return if any computation is considered active on a stream
     */
    public boolean isAnyComputationActive() { return !this.activeComputationsPerStream.isEmpty(); }

    protected void addActiveComputation(ExecutionDAG.DAGVertex vertex) {
        CUDAStream stream = vertex.getComputation().getStream();
        // Start tracking the stream if it wasn't already tracked;
        if (!activeComputationsPerStream.containsKey(stream)) {
            activeComputationsPerStream.put(stream, new HashSet<>());
        }
        // Associate the computation to the stream;
        activeComputationsPerStream.get(stream).add(vertex);
    }

    /**
     * Reset the association between streams and computations. All computations are finished, and all streams are free;
     */
    protected void resetActiveComputationState() {
        activeComputationsPerStream.keySet().forEach(s ->
            activeComputationsPerStream.get(s).forEach(v -> v.getComputation().setComputationFinished())
        );
        // Streams don't have any active computation;
        activeComputationsPerStream.clear();
        // All streams are free;
        retrieveNewStream.update(streams);
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        streams.forEach(runtime::cudaStreamDestroy);
        activeComputationsPerStream.clear();
        retrieveNewStream.cleanup();
        streams.clear();
    }

    /**
     * By default, create a new stream every time;
     */
    private class AlwaysNewRetrieveStream extends RetrieveNewStream {

        @Override
        public CUDAStream retrieve() {
            return createStream();
        }
    }

    /**
     * Keep a queue of free (currently not utilized) streams, and retrieve the oldest one added to the queue;
     */
    private class FifoRetrieveStream extends RetrieveNewStream {

        /**
         * Keep a queue of free streams;
         */
        private final Queue<CUDAStream> freeStreams = new ArrayDeque<>();
        /**
         * Ensure that streams in the queue are always unique;
         */
        private final Set<CUDAStream> uniqueFreeStreams = new HashSet<>();

        @Override
        void update(CUDAStream stream) {
            if (!uniqueFreeStreams.contains(stream)) {
                freeStreams.add(stream);
                uniqueFreeStreams.add(stream);
            }
        }

        @Override
        void update(Collection<CUDAStream> streams) {
            Set<CUDAStream> newStreams = streams.stream().filter(s -> !freeStreams.contains(s)).collect(Collectors.toSet());
            freeStreams.addAll(newStreams);
            uniqueFreeStreams.addAll(newStreams);
        }

        @Override
        CUDAStream retrieve() {
            if (freeStreams.isEmpty()) {
                // Create a new stream if none is available;
                return createStream();
            } else {
                // Get the first stream available, and remove it from the list of free streams;
                CUDAStream stream = freeStreams.poll();
                uniqueFreeStreams.remove(stream);
                return stream;
            }
        }
    }

    /**
     * By default, use the same stream as the parent computation;
     */
    private static class DefaultRetrieveParentStream extends RetrieveParentStream {

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            return vertex.getParentComputations().get(0).getStream();
        }
    }

    /**
     * If a vertex has more than one children, each children is independent (otherwise the dependency would be added
     * from one children to the other, and not from the actual parent).
     * As such, children can be executed on different streams. In practice, this situation happens when children
     * depends on disjoint arguments subsets of the parent kernel, e.g. K1(X,Y), K2(X), K3(Y).
     * This policy re-uses the parent(s) stream(s) when possible,
     * and computes other streams using the current {@link RetrieveNewStream};
     */
    private static class DisjointRetrieveParentStream extends RetrieveParentStream {
        private final RetrieveNewStream retrieveNewStream;

        // Keep track of computations for which we have already re-used the stream;
        private final Set<ExecutionDAG.DAGVertex> reusedComputations = new HashSet<>();

        public DisjointRetrieveParentStream(RetrieveNewStream retrieveNewStream) {
            this.retrieveNewStream = retrieveNewStream;
        }

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            // Keep only parent vertices for which we haven't reused the stream yet;
            List<ExecutionDAG.DAGVertex> availableParents = vertex.getParentVertices().stream()
                    .filter(v -> !reusedComputations.contains(v))
                    .collect(Collectors.toList());
            // If there is at least one stream that can be re-used, take it;
            if (!availableParents.isEmpty()) {
                // The computation cannot be considered again;
                reusedComputations.add(availableParents.get(0));
                // Return the stream associated to this computation;
                return availableParents.get(0).getComputation().getStream();
            } else {
                // If no parent stream can be reused, provide a new stream to this computation
                //   (or possibly a free one, depending on the policy);
                return retrieveNewStream.retrieve();
            }
        }
    }
}
