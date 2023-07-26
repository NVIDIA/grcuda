/*
 * Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.nvidia.grcuda.runtime.stream;

import com.nvidia.grcuda.CUDAEvent;
import com.nvidia.grcuda.GrCUDALogger;
import com.nvidia.grcuda.GrCUDAOptionMap;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.DeviceList;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.GrCUDAStreamPolicy;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.oracle.truffle.api.TruffleLogger;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

public class GrCUDAStreamManager {

    private static final TruffleLogger STREAM_LOGGER = GrCUDALogger.getLogger(GrCUDALogger.STREAM_LOGGER);
    
    /**
     * Reference to the CUDA runtime that manages the streams;
     */
    protected final CUDARuntime runtime;
    /**
     * Logging of kernel execution times option
     */
    protected final Boolean isTimeComputation;
    /**
     * Track the active computations each stream has, excluding the default stream;
     */
    protected final Map<CUDAStream, Set<ExecutionDAG.DAGVertex>> activeComputationsPerStream = new HashMap<>();

    /**
     * Handle for all the policies to assign streams and devices to a new computation that can run on CUDA stream;
     */
    private final GrCUDAStreamPolicy streamPolicy;
    
    public GrCUDAStreamManager(CUDARuntime runtime, GrCUDAOptionMap options) {
        this(runtime,
             options.getRetrieveNewStreamPolicy(),
             options.getRetrieveParentStreamPolicy(),
             options.getDeviceSelectionPolicy(),
             options.isTimeComputation(),
             options.getBandwidthMatrix(),
             options.getDataThreshold());
    }

    public GrCUDAStreamManager(
            CUDARuntime runtime,
            RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
            RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum,
            DeviceSelectionPolicyEnum deviceSelectionPolicyEnum,
            boolean isTimeComputation,
            String bandwidthMatrixPath,
            double dataThreshold) {
        this(runtime, isTimeComputation, new GrCUDAStreamPolicy(runtime, retrieveNewStreamPolicyEnum, retrieveParentStreamPolicyEnum, deviceSelectionPolicyEnum, bandwidthMatrixPath, dataThreshold));
    }

    public GrCUDAStreamManager(
            CUDARuntime runtime,
            boolean isTimeComputation,
            GrCUDAStreamPolicy streamPolicy) {
        this.runtime = runtime;
        this.isTimeComputation = isTimeComputation;
        this.streamPolicy = streamPolicy;
    }

    /**
     * Assign a {@link CUDAStream} to the input computation, based on its dependencies and on the available streams.
     * This function has no effect if the stream was manually specified by the user;
     * @param vertex an input computation for which we want to assign a stream
     */
    public void assignStream(ExecutionDAG.DAGVertex vertex) {
        // If the computation cannot use customized streams, return immediately;
        if (vertex.getComputation().canUseStream()) {
            // Else, obtain the stream (and the GPU device) for this computation from the stream policy manager;
            CUDAStream stream = this.streamPolicy.retrieveStream(vertex);
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
     * and can be used to time the execution of the computation;
     * @param vertex an input computation for which we want to assign an event
     */
    public void assignEventStart(ExecutionDAG.DAGVertex vertex) {
        // If the computation cannot use customized streams, return immediately;
        if (isTimeComputation && vertex.getComputation().canUseStream()) {
            // cudaEventRecord is sensitive to the ctx of the device that is currently set, so we call cudaSetDevice;
            runtime.cudaSetDevice(vertex.getComputation().getStream().getStreamDeviceId());
            CUDAEvent event = runtime.cudaEventCreate();
            runtime.cudaEventRecord(event, vertex.getComputation().getStream());
            vertex.getComputation().setEventStart(event);
        }
    }

    /**
     * Associate a new {@link CUDAEvent} to this computation, if the computation is done on a {@link CUDAStream}.
     * The event is created and recorded on the stream where the computation is running,
     * and can be used for precise synchronization of children computation;
     * @param vertex an input computation for which we want to assign an event
     */
    public void assignEventStop(ExecutionDAG.DAGVertex vertex) {
        // If the computation cannot use customized streams, return immediately;
        if (vertex.getComputation().canUseStream()) {
            CUDAEvent event = runtime.cudaEventCreate();
            runtime.cudaEventRecord(event, vertex.getComputation().getStream());
            vertex.getComputation().setEventStop(event);
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
                        STREAM_LOGGER.finest(() -> "--\tsync stream " + stream + " by " + vertex.getComputation());
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
                if (parent.getEventStop().isPresent()) {
                    CUDAEvent event = parent.getEventStop().get();
                    runtime.cudaStreamWaitEvent(vertex.getComputation().getStream(), event);

                    STREAM_LOGGER.finest(() -> "\t* wait event on stream; stream to sync=" + stream.getStreamNumber()
                            + "; stream that waits=" + vertex.getComputation().getStream().getStreamNumber()
                            + "; event=" + event.getEventNumber());
                } else {
                    STREAM_LOGGER.warning(() -> "\t* missing event to sync child computation=" + vertex.getComputation() +
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
            STREAM_LOGGER.finest(() -> "--\tsync stream=" + s.getStreamNumber() + " by " + vertex.getComputation());
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
            this.streamPolicy.updateNewStreamRetrieval(s);
        });
    }

    protected void setComputationFinishedInner(GrCUDAComputationalElement computation) {
        computation.setComputationFinished();
        if (computation.getEventStop().isPresent()) {
            if (isTimeComputation && computation.getEventStart().isPresent()) {
                // Switch to the device where the computation has been done, otherwise we cannot call the cudaEventElapsedTime API;
                runtime.cudaSetDevice(computation.getStream().getStreamDeviceId());
                float timeMilliseconds = runtime.cudaEventElapsedTime(computation.getEventStart().get(), computation.getEventStop().get());
                computation.setExecutionTime(timeMilliseconds);
                // Destroy the start event associated to this computation:
                runtime.cudaEventDestroy(computation.getEventStart().get());
            }
            // Destroy the stop event associated to this computation:
            runtime.cudaEventDestroy(computation.getEventStop().get());

        } else {
            STREAM_LOGGER.warning(() -> "missing event to destroy for computation=" + computation);
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
                    this.streamPolicy.updateNewStreamRetrieval(stream);
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
        return this.streamPolicy.getNumberOfStreams();
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
        this.streamPolicy.updateNewStreamRetrieval();
    }

    public DeviceList getDeviceList() {
        return this.streamPolicy.getDevicesManager().getDeviceList();
    }

    public Device getDevice(int deviceId) {
        return this.streamPolicy.getDevicesManager().getDevice(deviceId);
    }

    public GrCUDAStreamPolicy getStreamPolicy() {
        return streamPolicy;
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        this.activeComputationsPerStream.clear();
        this.streamPolicy.cleanup();
    }
}
