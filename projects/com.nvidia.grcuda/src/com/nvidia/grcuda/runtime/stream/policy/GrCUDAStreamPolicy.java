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
                this.deviceSelectionPolicy = new MinMinTransferTimeDeviceSelectionPolicy(devicesManager, dataThreshold, bandwidthMatrixPath);
                break;
            case MINMAX_TRANSFER_TIME:
                this.deviceSelectionPolicy = new MinMaxTransferTimeDeviceSelectionPolicy(devicesManager, dataThreshold, bandwidthMatrixPath);
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

    /////////////////////////////////////////////////////////////
    // List of interfaces that implement DeviceSelectionPolicy //
    /////////////////////////////////////////////////////////////

    /**
     * With some policies (e.g. the ones that don't support multiple GPUs), we never have to perform device selection.
     * Simply return the currently active device;
     */
    public static class SingleDeviceSelectionPolicy extends DeviceSelectionPolicy {
        public SingleDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
            super(devicesManager);
        }

        @Override
        public Device retrieve(ExecutionDAG.DAGVertex vertex) {
            return devicesManager.getCurrentGPU();
        }

        @Override
        Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            // There's only one device available, anyway;
            return this.retrieve(vertex);
        }
    }

    /**
     * Basic class for multi-GPU scheduling. Simply rotate between all the available device.
     * Not recommended for real utilization, but it can be useful for debugging
     * or as fallback for more complex policies.
     */
    public static class RoundRobinDeviceSelectionPolicy extends DeviceSelectionPolicy {
        private int nextDevice = 0;

        public RoundRobinDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
            super(devicesManager);
        }

        private void increaseNextDevice(int startDevice) {
            this.nextDevice = (startDevice + 1) % this.devicesManager.getNumberOfGPUsToUse();
        }

        public int getInternalState() {
            return nextDevice;
        }

        @Override
        public Device retrieve(ExecutionDAG.DAGVertex vertex) {
            Device device = this.devicesManager.getDevice(nextDevice);
            increaseNextDevice(nextDevice);
            return device;
        }

        @Override
        Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            // Sort the devices by ID;
            List<Device> sortedDevices = devices.stream().sorted(Comparator.comparingInt(Device::getDeviceId)).collect(Collectors.toList());
            // Keep increasing the internal state, but make sure that the retrieved device is among the ones in the input list;
            Device device = sortedDevices.get(nextDevice % devices.size());
            increaseNextDevice(nextDevice);
            return device;
        }
    }

    /**
     * We assign computations to the device with fewer active streams.
     * A stream is active if there's any computation assigned to it that has not been flagged as "finished".
     * The idea is to keep all devices equally busy, and avoid having devices that are used less than others;
     */
    public static class StreamAwareDeviceSelectionPolicy extends DeviceSelectionPolicy {
        public StreamAwareDeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
            super(devicesManager);
        }

        @Override
        Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            return devicesManager.findDeviceWithFewerBusyStreams(devices);
        }
    }

    /**
     * Given a computation, select the device that needs the least amount of data transfer.
     * In other words, select the device that already has the maximum amount of bytes available,
     * considering the size of the input arrays.
     * For each input array, we look at the devices where the array is up to date, and give a "score"
     * to that device that is equal to the array size. Then, we pick the device with maximum score.
     * In case of ties, pick the device with lower ID.
     * We do not consider the CPU as a meaningful location, because computations cannot be scheduled on the CPU.
     * If the computation does not have any data already present on any device,
     * choose the device with round-robin selection (using {@link RoundRobinDeviceSelectionPolicy};
     */
    public static class MinimizeTransferSizeDeviceSelectionPolicy extends DeviceSelectionPolicy {

        /**
         * Some policies can use a threshold that specifies how much data (in percentage) must be available
         * on a device so that the device can be considered for execution.
         * A low threshold favors exploitation (using the same device for most computations),
         * while a high threshold favors exploration (distribute the computations on different devices
         * even if some device would have slightly lower synchronization time);
         */
        protected final double dataThreshold;

        /**
         * Fallback policy in case no GPU has any up-tp-date data. We assume that for any GPU, transferring all the data
         * from the CPU would have the same cost, so we use this policy as tie-breaker;
         */
        RoundRobinDeviceSelectionPolicy roundRobin = new RoundRobinDeviceSelectionPolicy(devicesManager);

        public MinimizeTransferSizeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager, double dataThreshsold) {
            super(devicesManager);
            this.dataThreshold = dataThreshsold;
        }

        /**
         * For each input array of the computation, compute if the array is available on other devices and does not need to be
         * transferred. We track the total size, in bytes, that is already present on each device;
         * @param vertex the input computation
         * @param alreadyPresentDataSize the array where we store the size, in bytes, of data that is already present on each device.
         *                               The array must be zero-initialized and have size equal to the number of usable GPUs
         * @return if any data is present on any GPU. If false, we can use a fallback policy instead
         */
        boolean computeDataSizeOnDevices(ExecutionDAG.DAGVertex vertex, long[] alreadyPresentDataSize) {
            List<AbstractArray> arguments = vertex.getComputation().getArrayArguments();
            boolean isAnyDataPresentOnGPUs = false;  // True if there's at least a GPU with some data already available;
            for (AbstractArray a : arguments) {
                for (int location : a.getArrayUpToDateLocations()) {
                    if (location != CPUDevice.CPU_DEVICE_ID) {
                        alreadyPresentDataSize[location] += a.getSizeBytes();
                        isAnyDataPresentOnGPUs = true;
                    }
                }
            }
            return isAnyDataPresentOnGPUs;
        }

        /**
         * Find if any of the array inputs of the computation is present on the selected devices.
         * Used to understand if no device has any data already present, and the device selection policy
         * should fallback to a simpler device selection policy.
         * @param vertex the computation for which we want to select the device
         * @param devices the list of devices where the computation could be executed
         * @return if any of the computation's array inputs is already present on the specified devices
         */
        boolean isDataPresentOnGPUs(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            for (Device d : devices) {
                for (AbstractArray a : vertex.getComputation().getArrayArguments()) {
                    if (a.getArrayUpToDateLocations().contains(d.getDeviceId())) {
                        return true;
                    }
                }
            }
            return false;
        }

        /**
         * Find if any device has at least TRANSFER_THRESHOLD % of the size of data that is required by the computation;
         * @param alreadyPresentDataSize the size in bytes that is available on each device.
         *                              The array must contain all devices in the system, not just a subset of them
         * @param vertex the computation the we analyze
         * @param devices the list of devices considered by the function
         * @return if any device has at least RANSFER_THRESHOLD % of required data
         */
        boolean findIfAnyDeviceHasEnoughData(long[] alreadyPresentDataSize, ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            // Total size of the input arguments;
            long totalSize = vertex.getComputation().getArrayArguments().stream().map(AbstractArray::getSizeBytes).reduce(0L, Long::sum);
            // True if at least one device already has at least X% of the data required by the computation;
            for (Device d : devices) {
                if ((double) alreadyPresentDataSize[d.getDeviceId()] / totalSize > dataThreshold) {
                    return true;
                }
            }
            return false;
        }

        /**
         * Find the device with the most bytes in it. It's just an argmax on "alreadyPresentDataSize",
         * returning the device whose ID correspond to the maximum's index
         * @param devices the list of devices to consider for the argmax
         * @param alreadyPresentDataSize the array where we store the size, in bytes, of data that is already present on each device.
         *                               the array must be zero-initialized and have size equal to the number of usable GPUs
         * @return the device with the most data
         */
        private Device selectDeviceWithMostData(List<Device> devices, long[] alreadyPresentDataSize) {
            // Find device with maximum available data;
            Device deviceWithMaximumAvailableData = devices.get(0);
            for (Device d : devices) {
                if (alreadyPresentDataSize[d.getDeviceId()] > alreadyPresentDataSize[deviceWithMaximumAvailableData.getDeviceId()]) {
                    deviceWithMaximumAvailableData = d;
                }
            }
            return deviceWithMaximumAvailableData;
        }

        @Override
        public Device retrieve(ExecutionDAG.DAGVertex vertex) {
            // Array that tracks the size, in bytes, of data that is already present on each device;
            long[] alreadyPresentDataSize = new long[devicesManager.getNumberOfGPUsToUse()];
            // Compute the amount of data on each device, and if any device has any data at all;
            boolean isAnyDataPresentOnGPUs = computeDataSizeOnDevices(vertex, alreadyPresentDataSize);
            // If not device has at least X% of data available, it's not worth optimizing data locality (exploration preferred to exploitation);
            if (isAnyDataPresentOnGPUs && findIfAnyDeviceHasEnoughData(alreadyPresentDataSize, vertex, devicesManager.getUsableDevices())) {
                // Find device with maximum available data;
                return selectDeviceWithMostData(devicesManager.getUsableDevices(), alreadyPresentDataSize);
            } else {
                // No data is present on any GPU: select the device with round-robin;
                return roundRobin.retrieve(vertex);
            }
        }

        @Override
        Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            // Array that tracks the size, in bytes, of data that is already present on each device;
            long[] alreadyPresentDataSize = new long[devicesManager.getNumberOfGPUsToUse()];
            // Compute the amount of data on each device, and if any device has any data at all;
            computeDataSizeOnDevices(vertex, alreadyPresentDataSize);
            // If not device has at least X% of data available, it's not worth optimizing data locality (exploration preferred to exploitation);
            if (findIfAnyDeviceHasEnoughData(alreadyPresentDataSize, vertex, devices)) {
                // Find device with maximum available data;
                return selectDeviceWithMostData(devices, alreadyPresentDataSize);
            } else {
                // No data is present on any GPU: select the device with round-robin;
                return roundRobin.retrieve(vertex, devices);
            }
        }
    }

    /**
     * Given a computation, select the device that requires the least time to transfer data to it.
     * Compared to {@link MinimizeTransferSizeDeviceSelectionPolicy} this policy does not simply select the
     * device that requires the least data to be transferred to it, but also estimate the time that it takes
     * to transfer the data, given a heterogeneous multi-GPU system.
     * Given the complexity of CUDA's unified memory heuristics, we allow different heuristics to be used to estimate
     * the actual transfer speed, e.g. take the min or max possible values;
     * The speed of each GPU-GPU and CPU-GPU link is assumed to be stored in a file located in "$GRCUDA_HOME/grcuda_data/datasets/connection_graph/connection_graph.csv".
     * This file is generated as "cd $GRCUDA_HOME/projects/resources/cuda", "make connection_graph", "bin/connection_graph";
     */
    public abstract static class TransferTimeDeviceSelectionPolicy extends MinimizeTransferSizeDeviceSelectionPolicy {

        /**
         * This functional tells how the transfer bandwidth for some array and device is computed.
         * It should be max, min, mean, etc.;
         */
        private final java.util.function.BiFunction<Double, Double, Double> reduction;
        /**
         * Starting value of the reduction. E.g. it can be 0 if using max or mean, +inf if using min, etc.
         */
        private final double startValue;

        private final double[][] linkBandwidth = new double[devicesManager.getNumberOfGPUsToUse() + 1][devicesManager.getNumberOfGPUsToUse() + 1];

        public TransferTimeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager, double dataThreshold, String bandwidthMatrixPath, java.util.function.BiFunction<Double, Double, Double> reduction, double startValue) {
            super(devicesManager, dataThreshold);
            this.reduction = reduction;
            this.startValue = startValue;

            List<List<String>> records = new ArrayList<>();
            // Read each line in the CSV and store each line into a string array, splitting strings on ",";
            try (BufferedReader br = new BufferedReader(new FileReader(bandwidthMatrixPath))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(",");
                    records.add(Arrays.asList(values));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            // Read each line, and reconstruct the bandwidth matrix.
            // Given N GPUs and 1 CPU, we have a [GPU + 1][GPU+ 1] symmetric matrix.
            // Each line is "start_id", "end_id", "bandwidth";
            for (int il = 1; il < records.size(); il++) {
                int startDevice = Integer.parseInt(records.get(il).get(0));
                int endDevice = Integer.parseInt(records.get(il).get(1));
                // Skip invalid entries, and ignore GPUs with ID larger than the number of GPUs to use;
                if (startDevice >= -1 && startDevice < devicesManager.getNumberOfGPUsToUse()
                        && endDevice >= -1 && endDevice < devicesManager.getNumberOfGPUsToUse()) {
                    // Approximate to the floor, to smooth random bandwidth fluctuations in data transfer;
                    double bandwidth = Math.floor(Double.parseDouble(records.get(il).get(2)));
                    if (startDevice != -1) {
                        // GPU-GPU interconnection;
                        this.linkBandwidth[startDevice][endDevice] = bandwidth;
                    } else {
                        // -1 identifies CPU-GPU interconnection, store it in the last spot;
                        this.linkBandwidth[devicesManager.getNumberOfGPUsToUse()][endDevice] = bandwidth;
                        this.linkBandwidth[endDevice][devicesManager.getNumberOfGPUsToUse()] = bandwidth;
                    }
                }
            }
            // Interconnections are supposedly symmetric. Enforce this behavior by averaging results.
            // In other words, B[i][j] = B[j][j] <- (B[i][j] + B[j][i]) / 2.
            // Ignore the diagonal, and the last column and row (it represents the CPU and is already symmetric by construction);
            for (int i = 0; i < this.linkBandwidth.length - 1; i++) {
                for (int j = i; j < this.linkBandwidth.length - 1; j++) {
                    double averageBandwidth = (this.linkBandwidth[i][j] + this.linkBandwidth[j][i]) / 2;
                    this.linkBandwidth[i][j] = averageBandwidth;
                    this.linkBandwidth[j][i] = averageBandwidth;
                }
            }
        }

        /**
         * Estimate the bandwidth to transfer data to a "targetDevice" GPU, assuming
         * that data can be transferred from devices with index in "upToDateLocations".
         * @param targetDevice where we want to transfer data
         * @param upToDateLocations from where we can transfer data
         * @return an estimate of the transfer bandwidth
         */
        public double computeBandwidth(int targetDevice, Set<Integer> upToDateLocations) {
            // Hypotheses: we consider the max bandwidth towards the device targetDevice.
            // Initialization: min possible value, bandwidth = 0 GB/sec;
            double bandwidth = startValue;
            // Check that data is updated at least in some location. This is a precondition that must hold;
            if (upToDateLocations == null || upToDateLocations.isEmpty()) {
                throw new IllegalStateException("data is not updated in any location, when estimating bandwidth for device=" + targetDevice);
            }
            // If array a already present in device targetDevice, the transfer bandwidth to it is infinity.
            // We don't need to transfer it, so its transfer time will be 0;
            if (upToDateLocations.contains(targetDevice)) {
                bandwidth = Double.POSITIVE_INFINITY;
            } else {
                // Otherwise we consider the bandwidth to move array a to device targetDevice,
                // from each possible location containing the array a;
                for (int location : upToDateLocations) {
                    // The CPU bandwidth is stored in the last column;
                    int fromDevice = location != CPUDevice.CPU_DEVICE_ID ? location : linkBandwidth.length - 1;
                    // The matrix is symmetric, loading [targetDevice][fromDevice] is faster than [fromDevice][targetDevice];
                    bandwidth = reduction.apply(linkBandwidth[targetDevice][fromDevice], bandwidth);
                }
            }
            return bandwidth;
        }

        /**
         * For each device, measure how long it takes to transfer the data that is required
         * to run the computation in vertex
         * @param vertex the computation that we want to run
         * @param argumentTransferTime the array where we store the time, in seconds, to transfer the required data on each device
         *                             The array must be zero-initialized and have size equal to the number of usable GPUs
         * @return if any data is present on any GPU. If false, we can use a fallback policy instead
         */
        private boolean computeTransferTimes(ExecutionDAG.DAGVertex vertex, double[] argumentTransferTime) {
            List<AbstractArray> arguments = vertex.getComputation().getArrayArguments();

            // True if there's at least a GPU with some data already available;
            boolean isAnyDataPresentOnGPUs = false;

            // For each input array, consider how much time it takes to transfer it from every other device;
            for (AbstractArray a : arguments) {
                Set<Integer> upToDateLocations = a.getArrayUpToDateLocations();
                if (upToDateLocations.size() > 1 || (upToDateLocations.size() == 1 && !upToDateLocations.contains(CPUDevice.CPU_DEVICE_ID))) {
                    isAnyDataPresentOnGPUs = true;
                }
                // Check all available GPUs and compute the tentative transfer time for each of them.
                // to find the device where transfer time is minimum;
                for (int i = 0; i < argumentTransferTime.length; i++) {
                    // Add estimated transfer time;
                    argumentTransferTime[i] += a.getSizeBytes() / computeBandwidth(i, upToDateLocations);
                }
            }
            return isAnyDataPresentOnGPUs;
        }

        /**
         * Find the devices with at least TRANSFER_THRESHOLD % of the size of data that is required by the computation;
         * @param alreadyPresentDataSize the size in bytes that is available on each device.
         *                              The array must contain all devices in the system, not just a subset of them
         * @param vertex the computation the we analyze
         * @param devices the list of devices considered by the function
         * @return the list of devices that has at least TRANSFER_THRESHOLD % of required data
         */
        List<Device> findDevicesWithEnoughData(long[] alreadyPresentDataSize, ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            // List of devices with enough data;
            List<Device> devicesWithEnoughData = new ArrayList<>();
            // Total size of the input arguments;
            long totalSize = vertex.getComputation().getArrayArguments().stream().map(AbstractArray::getSizeBytes).reduce(0L, Long::sum);
            // True if at least one device already has at least X% of the data required by the computation;
            for (Device d : devices) {
                if ((double) alreadyPresentDataSize[d.getDeviceId()] / totalSize > dataThreshold) {
                    devicesWithEnoughData.add(d);
                }
            }
            return devicesWithEnoughData;
        }

        /**
         * Find the device with the lower synchronization time. It's just an argmin on "argumentTransferTime",
         * returning the device whose ID correspond to the minimum's index
         * @param devices the list of devices to consider for the argmin
         * @param argumentTransferTime the array where we store the time, in seconds, to transfer the required data on each device
         *                             The array must be zero-initialized and have size equal to the number of usable GPUs
         * @return the device with the most data
         */
        private Device findDeviceWithLowestTransferTime(List<Device> devices, double[] argumentTransferTime) {
            // The best device is the one with minimum transfer time;
            Device deviceWithMinimumTransferTime = devices.get(0);
            for (Device d : devices) {
                if (argumentTransferTime[d.getDeviceId()] < argumentTransferTime[deviceWithMinimumTransferTime.getDeviceId()]) {
                    deviceWithMinimumTransferTime = d;
                }
            }
            return deviceWithMinimumTransferTime;
        }

        @Override
        public Device retrieve(ExecutionDAG.DAGVertex vertex) {
            return this.retrieveImpl(vertex, devicesManager.getUsableDevices());
        }

        @Override
        Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
            // Estimated transfer time if the computation is scheduled on device i-th;
            double[] argumentTransferTime = new double[devicesManager.getNumberOfGPUsToUse()];
            // Compute the synchronization time on each device, and if any device has any data at all;
            boolean isAnyDataPresentOnGPUs = computeTransferTimes(vertex, argumentTransferTime);
            List<Device> devicesWithEnoughData = new ArrayList<>();
            if (isAnyDataPresentOnGPUs) {  // Skip this step if no GPU has any data in it;
                // Array that tracks the size, in bytes, of data that is already present on each device;
                long[] alreadyPresentDataSize = new long[devicesManager.getNumberOfGPUsToUse()];
                // Compute the amount of data on each device, and if any device has any data at all;
                computeDataSizeOnDevices(vertex, alreadyPresentDataSize);
                // Compute the list of devices that have at least X% of data already available;
                devicesWithEnoughData = findDevicesWithEnoughData(alreadyPresentDataSize, vertex, devices);
            }
            // If no device has at least X% of data available, it's not worth optimizing data locality (exploration preferred to exploitation);
            if (isAnyDataPresentOnGPUs && !devicesWithEnoughData.isEmpty()) {
                // The best device is the one with minimum transfer time;
                return findDeviceWithLowestTransferTime(devicesWithEnoughData, argumentTransferTime);
            } else {
                // No data is present on any GPU: select the device with round-robin;
                return roundRobin.retrieve(vertex, devices);
            }
        }

        public double[][] getLinkBandwidth() {
            return linkBandwidth;
        }
    }

    /**
     * Assume that data are transferred from the device that gives the best possible bandwidth.
     * In other words, find the minimum transfer time among all devices considering the minimum transfer time for each device;
     */
    public class MinMinTransferTimeDeviceSelectionPolicy extends TransferTimeDeviceSelectionPolicy {
        public MinMinTransferTimeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager, double dataThreshold, String bandwidthMatrixPath) {
            // Use max, we pick the maximum bandwidth between two devices;
            super(devicesManager, dataThreshold, bandwidthMatrixPath, Math::max, 0);
        }
    }

    /**
     * Assume that data are transferred from the device that gives the worst possible bandwidth.
     * In other words, find the minimum transfer time among all devices considering the maximum transfer time for each device;
     */
    public class MinMaxTransferTimeDeviceSelectionPolicy extends TransferTimeDeviceSelectionPolicy {
        public MinMaxTransferTimeDeviceSelectionPolicy(GrCUDADevicesManager devicesManager, double dataThreshold, String bandwidthMatrixPath) {
            // Use min, we pick the minimum bandwidth between two devices;
            super(devicesManager, dataThreshold, bandwidthMatrixPath, Math::min, Double.POSITIVE_INFINITY);
        }
    }
}
