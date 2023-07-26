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
package com.nvidia.grcuda.runtime.stream.policy;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDALogger;
import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.oracle.truffle.api.TruffleLogger;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GrCUDAStreamPolicy {

    /**
     * Reference to the class that manages the GPU devices in this system;
     */
    protected final GrCUDADevicesManager devicesManager;
    /**
     * Total number of CUDA streams created so far;
     */
    private int totalNumberOfStreams = 0;

    private final RetrieveNewStreamPolicy retrieveNewStreamPolicy;
    private final RetrieveParentStreamPolicy retrieveParentStreamPolicy;
    protected final DeviceSelectionPolicy deviceSelectionPolicy;

    private static final TruffleLogger STREAM_LOGGER = GrCUDALogger.getLogger(GrCUDALogger.STREAM_LOGGER);

    public GrCUDAStreamPolicy(GrCUDADevicesManager devicesManager,
                              RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
                              RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum,
                              DeviceSelectionPolicyEnum deviceSelectionPolicyEnum,
                              String bandwidthMatrixPath,
                              double dataThreshold) {
        this.devicesManager = devicesManager;
        // When using a stream selection policy that supports multiple GPUs,
        // we also need a policy to choose the device where the computation is executed;
        switch (deviceSelectionPolicyEnum) {
            case ROUND_ROBIN:
                this.deviceSelectionPolicy = new RoundRobinDeviceSelectionPolicy(devicesManager);
                break;
            case STREAM_AWARE:
                this.deviceSelectionPolicy = new StreamAwareDeviceSelectionPolicy(devicesManager);
                break;
            case MIN_TRANSFER_SIZE:
                this.deviceSelectionPolicy = new MinimizeTransferSizeDeviceSelectionPolicy(devicesManager, dataThreshold);
                break;
            case MINMIN_TRANSFER_TIME:
                this.deviceSelectionPolicy = new TransferTimeDeviceSelectionPolicy.MinMinTransferTimeDeviceSelectionPolicy(devicesManager, dataThreshold, bandwidthMatrixPath);
                break;
            case MINMAX_TRANSFER_TIME:
                this.deviceSelectionPolicy = new TransferTimeDeviceSelectionPolicy.MinMaxTransferTimeDeviceSelectionPolicy(devicesManager, dataThreshold, bandwidthMatrixPath);
                break;
            default:
                STREAM_LOGGER.finer("Disabled device selection policy, it is not necessary to use one as retrieveParentStreamPolicyEnum=" + retrieveParentStreamPolicyEnum);
                this.deviceSelectionPolicy = new SingleDeviceSelectionPolicy(devicesManager);
        }
        // Get how streams are retrieved for computations without parents;
        switch (retrieveNewStreamPolicyEnum) {
            case REUSE:
                this.retrieveNewStreamPolicy = new ReuseRetrieveStreamPolicy(this.deviceSelectionPolicy);
                break;
            case ALWAYS_NEW:
                this.retrieveNewStreamPolicy = new AlwaysNewRetrieveStreamPolicy(this.deviceSelectionPolicy);
                break;
            default:
                STREAM_LOGGER.severe("Cannot select a RetrieveNewStreamPolicy. The selected execution policy is not valid: " + retrieveNewStreamPolicyEnum);
                throw new GrCUDAException("selected RetrieveNewStreamPolicy is not valid: " + retrieveNewStreamPolicyEnum);
        }
        // Get how streams are retrieved for computations with parents;
        switch (retrieveParentStreamPolicyEnum) {
            case DISJOINT:
                this.retrieveParentStreamPolicy = new DisjointRetrieveParentStreamPolicy(this.retrieveNewStreamPolicy);
                break;
            case SAME_AS_PARENT:
                this.retrieveParentStreamPolicy = new SameAsParentRetrieveParentStreamPolicy();
                break;
            case MULTIGPU_EARLY_DISJOINT:
                this.retrieveParentStreamPolicy = new MultiGPUEarlySelectionDisjointRetrieveParentStreamPolicy(this.retrieveNewStreamPolicy, this.deviceSelectionPolicy);
                break;
            case MULTIGPU_DISJOINT:
                this.retrieveParentStreamPolicy = new MultiGPUDisjointRetrieveParentStreamPolicy(this.devicesManager, this.retrieveNewStreamPolicy, this.deviceSelectionPolicy);
                break;
            default:
                STREAM_LOGGER.severe("Cannot select a RetrieveParentStreamPolicy. The selected execution policy is not valid: " + retrieveParentStreamPolicyEnum);
                throw new GrCUDAException("selected RetrieveParentStreamPolicy is not valid: " + retrieveParentStreamPolicyEnum);
        }
    }

    public GrCUDAStreamPolicy(CUDARuntime runtime,
                              RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
                              RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum,
                              DeviceSelectionPolicyEnum deviceSelectionPolicyEnum,
                              String bandwidthMatrixPath,
                              double dataThrehsold) {
        this(new GrCUDADevicesManager(runtime), retrieveNewStreamPolicyEnum, retrieveParentStreamPolicyEnum, deviceSelectionPolicyEnum, bandwidthMatrixPath, dataThrehsold);
    }

    /**
     * Create a new {@link CUDAStream} on the current device;
     */
    public CUDAStream createStream() {
        CUDAStream newStream = this.devicesManager.getCurrentGPU().createStream();
        this.totalNumberOfStreams++;
        return newStream;
    }

    /**
     * Create a new {@link CUDAStream} on the specified device;
     */
    public CUDAStream createStream(int gpu) {
        CUDAStream newStream = this.devicesManager.getDevice(gpu).createStream();
        this.totalNumberOfStreams++;
        return newStream;
    }

    /**
     * Obtain the stream on which to execute the input computation.
     * If the computation doesn't have any parent, obtain a new stream or a free stream.
     * If the computation has parents, we might be reuse the stream of one of the parents.
     * Each stream is uniquely associated to a single GPU. If using multiple GPUs,
     * the choice of the stream also implies the choice of the GPU where the computation is executed;
     *
     * @param vertex the input computation for which we choose a stream;
     * @return the stream on which we execute the computation
     */
    public CUDAStream retrieveStream(ExecutionDAG.DAGVertex vertex) {
        if (vertex.isStart()) {
            // If the computation doesn't have parents, provide a new stream to it.
            // When using multiple GPUs, also select the device;
            return retrieveNewStream(vertex);
        } else {
            // Else, compute the streams used by the parent computations.
            // When using multiple GPUs, we might want to select the device as well,
            // if multiple suitable parent streams are available;
            return retrieveParentStream(vertex);
        }
    }

    CUDAStream retrieveNewStream(ExecutionDAG.DAGVertex vertex) {
        return this.retrieveNewStreamPolicy.retrieve(vertex);
    }

    CUDAStream retrieveParentStream(ExecutionDAG.DAGVertex vertex) {
        return this.retrieveParentStreamPolicy.retrieve(vertex);
    }

    /**
     * Update the status of a single stream within the NewStreamRetrieval policy;
     *
     * @param stream a stream to update;
     */
    public void updateNewStreamRetrieval(CUDAStream stream) {
        this.retrieveNewStreamPolicy.update(stream);
    }

    /**
     * Update the status of all streams within the NewStreamRetrieval policy,
     * saying for example that all can be reused;
     */
    public void updateNewStreamRetrieval() {
        // All streams are free to be reused;
        this.retrieveNewStreamPolicy.update();
    }

    void cleanupNewStreamRetrieval() {
        this.retrieveNewStreamPolicy.cleanup();
    }

    /**
     * Obtain the number of streams created so far;
     */
    public int getNumberOfStreams() {
        return this.totalNumberOfStreams;
    }

    public GrCUDADevicesManager getDevicesManager() {
        return devicesManager;
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        this.cleanupNewStreamRetrieval();
        this.devicesManager.cleanup();
    }

    ///////////////////////////////////////////////////////////////
    // List of interfaces that implement RetrieveNewStreamPolicy //
    ///////////////////////////////////////////////////////////////

    /**
     * By default, create a new stream every time;
     */
    private class AlwaysNewRetrieveStreamPolicy extends RetrieveNewStreamPolicy {

        AlwaysNewRetrieveStreamPolicy(DeviceSelectionPolicy deviceSelectionPolicy) {
            super(deviceSelectionPolicy, GrCUDAStreamPolicy.this.devicesManager);
        }

        @Override
        CUDAStream retrieveStreamFromDevice(Device device) {
            return createStream(device.getDeviceId());
        }
    }

    /**
     * Keep a set of free (currently not utilized) streams, and retrieve one of them instead of always creating new streams;
     */
    private class ReuseRetrieveStreamPolicy extends RetrieveNewStreamPolicy {

        ReuseRetrieveStreamPolicy(DeviceSelectionPolicy deviceSelectionPolicy) {
            super(deviceSelectionPolicy, GrCUDAStreamPolicy.this.devicesManager);
        }

        @Override
        CUDAStream retrieveStreamFromDevice(Device device) {
            if (device.getNumberOfFreeStreams() == 0) {
                // Create a new stream if none is available;
                return createStream(device.getDeviceId());
            } else {
                return device.getFreeStream();
            }
        }
    }

    //////////////////////////////////////////////////////////////////
    // List of interfaces that implement RetrieveParentStreamPolicy //
    //////////////////////////////////////////////////////////////////

    /**
     * By default, use the same stream as the parent computation;
     */
    private static class SameAsParentRetrieveParentStreamPolicy extends RetrieveParentStreamPolicy {

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
     * and computes other streams using the current {@link RetrieveNewStreamPolicy};
     */
    private static class DisjointRetrieveParentStreamPolicy extends RetrieveParentStreamPolicy {
        protected final RetrieveNewStreamPolicy retrieveNewStreamPolicy;

        // Keep track of computations for which we have already re-used the stream;
        protected final Set<ExecutionDAG.DAGVertex> reusedComputations = new HashSet<>();

        public DisjointRetrieveParentStreamPolicy(RetrieveNewStreamPolicy retrieveNewStreamPolicy) {
            this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
        }

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            // Keep only parent vertices for which we haven't reused the stream yet;
            List<ExecutionDAG.DAGVertex> availableParents = vertex.getParentVertices().stream()
                    .filter(v -> !reusedComputations.contains(v)).collect(Collectors.toList());
            // If there is at least one stream that can be re-used, take it.
            // When using multiple devices, we just take a parent stream without considering the device of the parent;
            // FIXME: we might take a random parent. Or use round-robin;
            if (!availableParents.isEmpty()) {
                // The computation cannot be considered again;
                reusedComputations.add(availableParents.iterator().next());
                // Return the stream associated to this computation;
                return availableParents.iterator().next().getComputation().getStream();
            } else {
                // If no parent stream can be reused, provide a new stream to this computation
                //   (or possibly a free one, depending on the policy);
                return retrieveNewStreamPolicy.retrieve(vertex);
            }
        }
    }

    /**
     * This policy extends DisjointRetrieveParentStreamPolicy with multi-GPU support for reused streams,
     * not only for newly created streams.
     * In this policy, we first select the ideal GPU for the input computation.
     * Then, we find if any of the reusable streams is allocated on that device.
     * If not, we create a new stream on the ideal GPU;
     */
    private static class MultiGPUEarlySelectionDisjointRetrieveParentStreamPolicy extends DisjointRetrieveParentStreamPolicy {

        private final DeviceSelectionPolicy deviceSelectionPolicy;

        public MultiGPUEarlySelectionDisjointRetrieveParentStreamPolicy(RetrieveNewStreamPolicy retrieveNewStreamPolicy, DeviceSelectionPolicy deviceSelectionPolicy) {
            super(retrieveNewStreamPolicy);
            this.deviceSelectionPolicy = deviceSelectionPolicy;
        }

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            // Keep only parent vertices for which we haven't reused the stream yet;
            List<ExecutionDAG.DAGVertex> availableParents = vertex.getParentVertices().stream()
                    .filter(v -> !reusedComputations.contains(v))
                    .collect(Collectors.toList());
            // First, select the ideal device to execute this computation;
            Device selectedDevice = deviceSelectionPolicy.retrieve(vertex);

            // If at least one of the parents' streams is on the selected device, use that stream.
            // Otherwise, create a new stream on the selected device;
            if (!availableParents.isEmpty()) {
                for (ExecutionDAG.DAGVertex v : availableParents) {
                    if (v.getComputation().getStream().getStreamDeviceId() == selectedDevice.getDeviceId()) {
                        // We found a parent whose stream is on the selected device;
                        reusedComputations.add(v);
                        return v.getComputation().getStream();
                    }
                }
            }
            // If no parent stream can be reused, provide a new stream to this computation
            //   (or possibly a free one, depending on the policy);
            return retrieveNewStreamPolicy.retrieveStreamFromDevice(selectedDevice);
        }
    }

    /**
     * This policy extends DisjointRetrieveParentStreamPolicy with multi-GPU support for reused streams,
     * not only for newly created streams.
     * In this policy, we select the streams that can be reused from the computation's parents.
     * Then, we find which of the parent's devices is the best for the input computation.
     * If no stream can be reused, we select a new device and create a stream on it;
     */
    private static class MultiGPUDisjointRetrieveParentStreamPolicy extends DisjointRetrieveParentStreamPolicy {

        private final DeviceSelectionPolicy deviceSelectionPolicy;
        private final GrCUDADevicesManager devicesManager;

        public MultiGPUDisjointRetrieveParentStreamPolicy(GrCUDADevicesManager devicesManager, RetrieveNewStreamPolicy retrieveNewStreamPolicy, DeviceSelectionPolicy deviceSelectionPolicy) {
            super(retrieveNewStreamPolicy);
            this.devicesManager = devicesManager;
            this.deviceSelectionPolicy = deviceSelectionPolicy;
        }

        @Override
        public CUDAStream retrieve(ExecutionDAG.DAGVertex vertex) {
            // Keep only parent vertices for which we haven't reused the stream yet;
            List<ExecutionDAG.DAGVertex> availableParents = vertex.getParentVertices().stream()
                    .filter(v -> !reusedComputations.contains(v))
                    .collect(Collectors.toList());
            // Map each parent's device to the respective parent;
            Map<Device, ExecutionDAG.DAGVertex> deviceParentMap = availableParents
                    .stream().collect(Collectors.toMap(
                            v -> devicesManager.getDevice(v.getComputation().getStream().getStreamDeviceId()),
                            Function.identity(),
                            (x, y) -> x, // If two parents have the same device, use the first parent;
                            HashMap::new // Use hashmap;
                    ));
            // If there's at least one free stream on the parents' devices,
            // select the best device among the available parent devices.
            // If no stream is available, create a new stream on the best possible device;
            if (!availableParents.isEmpty()) {
                // First, select the best device among the ones available;
                Device selectedDevice = deviceSelectionPolicy.retrieve(vertex, new ArrayList<>(deviceParentMap.keySet()));
                ExecutionDAG.DAGVertex selectedParent = deviceParentMap.get(selectedDevice);
                // We found a parent whose stream is on the selected device;
                reusedComputations.add(selectedParent);
                return selectedParent.getComputation().getStream();
            }
            // If no parent stream can be reused, provide a new stream to this computation
            //   (or possibly a free one, depending on the policy);
            return retrieveNewStreamPolicy.retrieve(vertex);
        }
    }
}
