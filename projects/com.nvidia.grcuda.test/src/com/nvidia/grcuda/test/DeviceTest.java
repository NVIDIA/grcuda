/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
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
package com.nvidia.grcuda.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;

public class DeviceTest {

    @Test
    public void testDeviceCount() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceCount = ctx.eval("grcuda", "cudaGetDeviceCount()");
            assertTrue(deviceCount.isNumber());
            assertTrue(deviceCount.asInt() > 0);
        }
    }

    @Test
    public void testGetDevicesLengthsMatchesDeviceCount() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceCount = ctx.eval("grcuda", "cudaGetDeviceCount()");
            assertTrue(deviceCount.isNumber());
            assertTrue(deviceCount.asInt() > 0);
            Value devices = ctx.eval("grcuda", "getdevices()");
            assertEquals(deviceCount.asInt(), devices.getArraySize());
        }
    }

    @Test
    public void testGetDevicesMatchesAllGetDevice() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value devices = ctx.eval("grcuda", "getdevices()");
            Value getDevice = ctx.eval("grcuda", "getdevice");
            for (int i = 0; i < devices.getArraySize(); ++i) {
                Value deviceFromArray = devices.getArrayElement(i);
                Value deviceFromFunction = getDevice.execute(i);
                assertEquals(i, deviceFromArray.getMember("id").asInt());
                assertEquals(i, deviceFromFunction.getMember("id").asInt());
            }
        }
    }

    @Test
    public void testCanReadSomeDeviceProperties() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value devices = ctx.eval("grcuda", "getdevices()");
            for (int i = 0; i < devices.getArraySize(); ++i) {
                Value device = devices.getArrayElement(i);
                Value prop = device.getMember("properties");
                // Sanity tests on some of the properties
                // device name is a non-zero string
                assertTrue(prop.getMember("deviceName").asString().length() > 0);

                // compute capability is at least compute Kepler (3.0)
                assertTrue(prop.getMember("computeCapabilityMajor").asInt() >= 3);

                // there is at least one multiprocessors
                assertTrue(prop.getMember("multiProcessorCount").asInt() > 0);

                // there is some device memory
                assertTrue(prop.getMember("totalDeviceMemory").asLong() > 0L);
            }
        }
    }

    @Test
    public void testCanSelectDevice() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value devices = ctx.eval("grcuda", "getdevices()");
            if (devices.getArraySize() > 1) {
                Value firstDevice = devices.getArrayElement(0);
                Value secondDevice = devices.getArrayElement(1);
                secondDevice.invokeMember("setCurrent");
                assertFalse(firstDevice.invokeMember("isCurrent").asBoolean());
                assertTrue(secondDevice.invokeMember("isCurrent").asBoolean());

                firstDevice.invokeMember("setCurrent");
                assertTrue(firstDevice.invokeMember("isCurrent").asBoolean());
                assertFalse(secondDevice.invokeMember("isCurrent").asBoolean());
            } else {
                // only one device available
                Value device = devices.getArrayElement(0);
                device.invokeMember("setCurrent");
                assertTrue(device.invokeMember("isCurrent").asBoolean());
            }
        }
    }

    @Test
    public void testDeviceMemoryAllocationReducesReportedFreeMemory() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value device = ctx.eval("grcuda", "getdevice(0)");
            Value props = device.getMember("properties");
            device.invokeMember("setCurrent");
            long totalMemoryBefore = props.getMember("totalDeviceMemory").asLong();
            long freeMemoryBefore = props.getMember("freeDeviceMemory").asLong();
            assertTrue(freeMemoryBefore <= totalMemoryBefore);

            // allocate memory on device (unmanaged)
            long arraySizeBytes = freeMemoryBefore / 3;
            Value cudaMalloc = ctx.eval("grcuda", "cudaMalloc");
            Value cudaFree = ctx.eval("grcuda", "cudaFree");
            Value gpuPointer = null;
            try {
                gpuPointer = cudaMalloc.execute(arraySizeBytes);
                // After allocation total memory must be the same as before but
                // the free memory must be lower by at least the amount of allocated bytes.
                long totalMemoryAfter = props.getMember("totalDeviceMemory").asLong();
                long freeMemoryAfter = props.getMember("freeDeviceMemory").asLong();
                assertEquals(totalMemoryBefore, totalMemoryAfter);
                assertTrue(freeMemoryAfter <= (freeMemoryBefore - arraySizeBytes));
            } finally {
                if (gpuPointer != null) {
                    cudaFree.execute(gpuPointer);
                }
            }
        }
    }

}
