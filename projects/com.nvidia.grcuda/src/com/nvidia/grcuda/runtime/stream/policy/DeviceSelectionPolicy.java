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
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

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
            // Sort the devices by ID;
            List<Device> sortedDevices = devices.stream().sorted(Comparator.comparingInt(Device::getDeviceId)).collect(Collectors.toList());
            return this.retrieveImpl(vertex, sortedDevices);
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
