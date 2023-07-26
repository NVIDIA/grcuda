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

import java.util.Collection;
import java.util.List;

import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.DeviceList;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

public class GrCUDADevicesManager {

    private final CUDARuntime runtime;
    private final DeviceList deviceList;

    /**
     * Initialize the GrCUDADevicesManager, creating a DeviceList that tracks all available GPUs.
     * @param runtime reference to the CUDA runtime
     */
    public GrCUDADevicesManager(CUDARuntime runtime) {
        this(runtime, new DeviceList(runtime));
    }

    /**
     * Initialize the GrCUDADevicesManager, using an existing DeviceList that tracks all available GPUs;
     * @param runtime reference to the CUDA runtime
     * @param deviceList list of available devices
     */
    public GrCUDADevicesManager(CUDARuntime runtime, DeviceList deviceList) {
        this.runtime = runtime;
        this.deviceList = deviceList;
    }

    /**
     * Find the device with the lowest number of busy stream on it and returns it.
     * A stream is busy if there's any computation assigned to it that has not been flagged as "finished".
     * If multiple devices have the same number of free streams, return the first.
     * In this implementation, consider all usable devices;
     * @return the device with fewer busy streams
     */
    public Device findDeviceWithFewerBusyStreams(){
        return findDeviceWithFewerBusyStreams(getUsableDevices());
    }

    /**
     * Find the device with the lowest number of busy stream on it and returns it.
     * A stream is busy if there's any computation assigned to it that has not been flagged as "finished".
     * If multiple devices have the same number of free streams, return the first;
     * @param devices the list of devices to inspect
     * @return the device with fewer busy streams
     */
    public Device findDeviceWithFewerBusyStreams(List<Device> devices){
        Device device = devices.get(0);
        int min = device.getNumberOfBusyStreams();
        for (Device d : devices) {
            int numBusyStreams = d.getNumberOfBusyStreams();
            if (numBusyStreams < min) {
                min = numBusyStreams;
                device = d;
            }
        }
        return device;
    }

    public Device getCurrentGPU(){
        return this.getDevice(this.runtime.getCurrentGPU());
    }

    public int getNumberOfGPUsToUse(){
        return this.runtime.getNumberOfGPUsToUse();
    }

    public DeviceList getDeviceList() {
        return deviceList;
    }

    public List<Device> getUsableDevices() {
        return deviceList.getDevices().subList(0, this.getNumberOfGPUsToUse());
    }

    public Device getDevice(int deviceId) {
        return deviceList.getDevice(deviceId);
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup() {
        this.deviceList.cleanup();
    }
}