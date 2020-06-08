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
    protected final Map<CUDAStream, Set<GrCUDAComputationalElement>> activeComputationsPerStream = new HashMap<>();
    /**
     * Associate to each computational element its parents that are executed on different streams.
     * This is required to set these parent computations as finished when the stream of the child element is synchronized;
     */
    protected final Map<GrCUDAComputationalElement, Set<GrCUDAComputationalElement>> additionalComputationsToFinish = new HashMap<>();

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
            case DEFAULT:
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
                stream = retrieveNewStream.retrieve();
            } else {
                // Else, compute the streams used by the parent computations.
                stream = this.retrieveParentStream.retrieve(vertex);
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

    public void syncParentStreams(ExecutionDAG.DAGVertex vertex) {
        // If the vertex can be executed on a CUDA stream, use CUDA events,
        //   otherwise use stream/device synchronization to block the host until synchronization is done;
        if (vertex.getComputation().canUseStream()) {
            syncStreamsUsingEvents(vertex);
        } else {
            if (this.isAnyComputationActive()) {
                Set<CUDAStream> streamsToSync = getParentStreams(vertex.getParentComputations());
                Optional<CUDAStream> additionalStream = vertex.getComputation().additionalStreamDependency();
                if (additionalStream.isPresent()) {
                    CUDAStream stream = additionalStream.get();
                    // If we require synchronization on the default stream, perform it in a specialized way;
                    if (stream.isDefaultStream()) {
                        System.out.println("--\tsync stream " + stream + " by " + vertex.getComputation());
                        // Synchronize the device;
                        syncDevice();
                        // All computations are now finished;
                        resetActiveComputationState();
                    } else if (!streamsToSync.contains(stream)) {
                        // Else add the computations related to the additional streams to the set and sync it,
                        //   as long as the additional stream isn't the same as the one that we have to sync anyway;
                        System.out.println("--\tsyncing additional stream " + stream + "...");
                        streamsToSync.add(stream);
                        syncParentStreamsImpl(streamsToSync, vertex.getComputation());
                    } else {
                        syncParentStreamsImpl(streamsToSync, vertex.getComputation());
                    }
                } else {
                    syncParentStreamsImpl(streamsToSync, vertex.getComputation());
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
        Set<CUDAStream> streamSynchronized = new HashSet<>();
        for (GrCUDAComputationalElement parent : vertex.getParentComputations()) {
            CUDAStream stream = parent.getStream();
            // Skip synchronization on the same stream where the new computation is executed,
            //   as operations scheduled on a stream are executed in order;
            if (!vertex.getComputation().getStream().equals(stream)) {
                // Don't process the same stream twice, in any case;
                if (!streamSynchronized.contains(stream)) {
                    // Create a new synchronization event on the stream;
                    CUDAEvent event = runtime.cudaEventCreate();
                    runtime.cudaEventRecord(event, stream);
                    runtime.cudaStreamWaitEvent(vertex.getComputation().getStream(), event);

                    streamSynchronized.add(stream);
                    System.out.println("\t* wait event on stream; stream to sync=" + stream.getStreamNumber()
                            + "; stream that waits=" + vertex.getComputation().getStream().getStreamNumber()
                            + "; event=" + event.getEventNumber());
                }
                // Track that the current computation has a parent executed on a different stream;
                trackAdditionalComputationsToFinish(vertex.getComputation(), parent);
            }
        }
    }

    /**
     * Associate a parent computation to its child computation, so that we remember to set the parent computation as finished
     * once the child stream has been synchronized;
     * @param computation a computational element
     * @param parent a parent computational element executed on a different stream
     */
    protected void trackAdditionalComputationsToFinish(GrCUDAComputationalElement computation, GrCUDAComputationalElement parent) {
        if (additionalComputationsToFinish.containsKey(computation)) {
            additionalComputationsToFinish.get(computation).add(parent);
        } else {
            additionalComputationsToFinish.put(computation, new HashSet<>(Collections.singletonList(parent)));
        }
        // Forward-propagation of dependencies: if the parent already had additional computations to sync,
        //   a sync on the child computation should sync them too;
        if (additionalComputationsToFinish.containsKey(parent)) {
            additionalComputationsToFinish.get(computation).addAll(additionalComputationsToFinish.get(parent));
        }
    }

    protected void syncParentStreamsImpl(
            Set<CUDAStream> streamsToSync,
            GrCUDAComputationalElement computationThatSyncs) {

        // Synchronize streams;
        streamsToSync.forEach(s -> {
            System.out.println("--\tsync stream=" + s.getStreamNumber() + " by " + computationThatSyncs);
            syncStream(s);
        });
        // Book-keeping: all computations on the synchronized streams are guaranteed to be finished;
        streamsToSync.forEach(s -> {
            try {
                activeComputationsPerStream.get(s).forEach(c -> {
                    c.setComputationFinished();
                    setParentComputationsFinished(c);
                });
            } catch (NullPointerException e) {
                System.out.println("missing stream " + s.getStreamNumber() + " for sync");
            }
            // Now the stream is free to be re-used;
            activeComputationsPerStream.remove(s);
            retrieveNewStream.update(s);
        });
    }

    private void setParentComputationsFinished(GrCUDAComputationalElement c) {
        if (additionalComputationsToFinish.containsKey(c)) {
            additionalComputationsToFinish.get(c).forEach(parent -> {
                if (!parent.isComputationFinished()) {
                    parent.setComputationFinished();
                    CUDAStream stream = parent.getStream();
                    // Stop considering this computation as active on its stream;
                    activeComputationsPerStream.get(stream).remove(parent);
                    // If this stream doesn't have any computation associated to it, it's free to use;
                    if (activeComputationsPerStream.get(stream).isEmpty()) {
                        activeComputationsPerStream.remove(stream);
                        retrieveNewStream.update(stream);
                    }
                    additionalComputationsToFinish.remove(parent);
                }
            });
            additionalComputationsToFinish.remove(c);
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
        return activeComputationsPerStream.get(stream).size();
    }

    /**
     * Check if any computation is currently marked as active, and is running on a stream.
     * If so, scheduling of new computations is likely to require synchronizations of some sort;
     * @return if any computation is considered active on a stream
     */
    public boolean isAnyComputationActive() { return !this.activeComputationsPerStream.isEmpty(); }

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
     * Reset the association between streams and computations. All computations are finished, and all streams are free;
     */
    protected void resetActiveComputationState() {
        activeComputationsPerStream.keySet().forEach(s ->
            activeComputationsPerStream.get(s).forEach(GrCUDAComputationalElement::setComputationFinished)
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
