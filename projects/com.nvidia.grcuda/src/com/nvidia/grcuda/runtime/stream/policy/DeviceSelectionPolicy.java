package com.nvidia.grcuda.runtime.stream.policy;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;

import java.util.List;

/**
 * When using multiple GPUs, selecting the stream where a computation is executed implies
 * the selection of a GPU, as each stream is uniquely associated to a single GPU.
 * This abstract class defines how a {@link GrCUDAStreamPolicy}
 * selects a {@link com.nvidia.grcuda.runtime.Device} on which a {@link com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement}
 * is executed. Device selection is performed by {@link RetrieveNewStreamPolicy} (when creating a new stream)
 * and {@link RetrieveParentStreamPolicy} (when the parent's stream cannot be directly reused).
 * For example, we can select the device that requires the least data transfer.
 */
public abstract class DeviceSelectionPolicy {

    protected final GrCUDADevicesManager devicesManager;

    public DeviceSelectionPolicy(GrCUDADevicesManager devicesManager) {
        this.devicesManager = devicesManager;
    }

    /**
     * Select the device where the computation will be executed.
     * By default call {@link DeviceSelectionPolicy#retrieve(ExecutionDAG.DAGVertex, List)} on all devices,
     * but it can be overridden to provide optimized behavior for the case when no restriction on specific devices is needed;
     * @param vertex the computation for which we want to select the device
     * @return the chosen device for the computation
     */
    public Device retrieve(ExecutionDAG.DAGVertex vertex) {
        return retrieveImpl(vertex, devicesManager.getUsableDevices());
    }

    /**
     * Restrict the device selection to the specific set of devices;
     * @param vertex the computation for which we want to select the device
     * @param devices the list of devices where the computation could be executed
     * @return the chosen device for the computation
     */
    public Device retrieve(ExecutionDAG.DAGVertex vertex, List<Device> devices) {
        if (devices == null) {
            throw new NullPointerException("the list of devices where the computation can be executed is null");
        } else if (devices.size() == 0) {
            throw new GrCUDAException("the list of devices where the computation can be executed is empty");
        } else {
            return this.retrieveImpl(vertex, devices);
        }
    }

    /**
     * Internal implementation of {@link DeviceSelectionPolicy#retrieve(ExecutionDAG.DAGVertex, List)},
     * assuming that the list of devices contains at least one device;
     * @param vertex the computation for which we want to select the device
     * @param devices the list of devices where the computation could be executed
     * @return the chosen device for the computation
     */
    abstract Device retrieveImpl(ExecutionDAG.DAGVertex vertex, List<Device> devices);
}
