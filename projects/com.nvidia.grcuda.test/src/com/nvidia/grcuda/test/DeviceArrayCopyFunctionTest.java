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
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;
import com.nvidia.grcuda.gpu.LittleEndianNativeArrayView;
import com.nvidia.grcuda.gpu.OffheapMemory;

public class DeviceArrayCopyFunctionTest {

    @Test
    public void testDeviceArrayCopyFromOffheapMemory() {
        final int numElements = 1000;
        final int numBytesPerInt = 4;
        final int numBytes = numElements * numBytesPerInt;
        try (OffheapMemory hostMemory = new OffheapMemory(numBytes)) {
            // create off-heap host memory of integers: [1, 2, 3, 4, ..., 1000]
            LittleEndianNativeArrayView hostArray = hostMemory.getLittleEndianView();
            for (int i = 0; i < numElements; ++i) {
                hostArray.setInt(i, i + 1);
            }
            try (Context ctx = Context.newBuilder().allowAllAccess(true).build()) {
                // create DeviceArray and copy content from off-heap host memory into it
                Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
                Value deviceArray = createDeviceArray.execute("int", numElements);
                deviceArray.invokeMember("copyFrom", hostMemory.getPointer(), numElements);

                // Verify content of device array
                for (int i = 0; i < numElements; ++i) {
                    assertEquals(i + 1, deviceArray.getArrayElement(i).asInt());
                }
            }
        }
    }

    @Test
    public void testDeviceArrayCopyToOffheapMemory() {
        final int numElements = 1000;
        final int numBytesPerInt = 4;
        final int numBytes = numElements * numBytesPerInt;
        try (OffheapMemory hostMemory = new OffheapMemory(numBytes)) {
            // create off-heap host memory of integers and initialize all elements to zero.
            LittleEndianNativeArrayView hostArray = hostMemory.getLittleEndianView();
            for (int i = 0; i < numElements; ++i) {
                hostArray.setInt(i, i);
            }
            try (Context ctx = Context.newBuilder().allowAllAccess(true).build()) {
                // create DeviceArray and set its content [1, 2, 3, 4, ..., 1000]
                Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
                Value deviceArray = createDeviceArray.execute("int", numElements);
                for (int i = 0; i < numElements; ++i) {
                    deviceArray.setArrayElement(i, i + 1);
                }
                // copy content of device array to off-heap host memory
                deviceArray.invokeMember("copyTo", hostMemory.getPointer(), numElements);

                // Verify content of device array
                for (int i = 0; i < numElements; ++i) {
                    assertEquals(i + 1, hostArray.getInt(i));
                }
            }
        }
    }

    @Test
    public void testDeviceArrayCopyFromDeviceArray() {
        final int numElements = 1000;
        try (Context ctx = Context.newBuilder().allowAllAccess(true).build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                sourceDeviceArray.setArrayElement(i, i + 1);
            }
            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                destinationDeviceArray.setArrayElement(i, 0);
            }
            destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements);
            // Verify content of device array
            for (int i = 0; i < numElements; ++i) {
                assertEquals(i + 1, destinationDeviceArray.getArrayElement(i).asInt());
            }
        }
    }

    @Test
    public void testDeviceArrayCopyToDeviceArray() {
        final int numElements = 1000;
        try (Context ctx = Context.newBuilder().allowAllAccess(true).build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                sourceDeviceArray.setArrayElement(i, i + 1);
            }
            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                destinationDeviceArray.setArrayElement(i, 0);
            }
            sourceDeviceArray.invokeMember("copyTo", destinationDeviceArray, numElements);
            // Verify content of device array
            for (int i = 0; i < numElements; ++i) {
                assertEquals(i + 1, destinationDeviceArray.getArrayElement(i).asInt());
            }
        }
    }
}
