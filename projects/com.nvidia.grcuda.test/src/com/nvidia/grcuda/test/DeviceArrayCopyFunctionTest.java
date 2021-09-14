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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
            try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
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
            try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
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
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
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
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
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

    @Test
    public void testMultiDimDeviceArrayCopyFromDeviceArray() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);

            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);

            // Set each row to j;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, j);
                }
            }
            destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements1 * numElements2);
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyToDeviceArray() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            // Set each row to j;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, j);
                }
            }
            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);

            sourceDeviceArray.invokeMember("copyTo", destinationDeviceArray, numElements1 * numElements2);
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyToDeviceArrayRow() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            // Set each rows 3 to j;
            for (int j = 0; j < numElements2; ++j) {
                sourceDeviceArray.getArrayElement(3).setArrayElement(j, j);
            }

            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements2);

            sourceDeviceArray.getArrayElement(3).invokeMember("copyTo", destinationDeviceArray, sourceDeviceArray.getArrayElement(3).getArraySize());
            // Verify content of device array
            for (int j = 0; j < numElements2; ++j) {
                assertEquals(j, destinationDeviceArray.getArrayElement(j).asInt());
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyFromDeviceArrayRow() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements2);
            // Set each value to j;
            for (int j = 0; j < numElements2; ++j) {
                destinationDeviceArray.setArrayElement(j, j);
            }

            sourceDeviceArray.getArrayElement(3).invokeMember("copyFrom", destinationDeviceArray, sourceDeviceArray.getArrayElement(3).getArraySize());
            // Verify content of device array
            for (int j = 0; j < numElements2; ++j) {
                assertEquals(j, sourceDeviceArray.getArrayElement(3).getArrayElement(j).asInt());
            }
        }
    }

    // FIXME: memcpy on single dimensions of column-major arrays doesn't work, as memory is not contiguous;

    @Test
    public void testMultiDimDeviceArrayCopyFromDeviceArrayF() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");

            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");

            // Set each row to j;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, j);
                }
            }
            destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements1 * numElements2);
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyToDeviceArrayF() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");
            // Set each row to j;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, j);
                }
            }
            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");

            sourceDeviceArray.invokeMember("copyTo", destinationDeviceArray, numElements1 * numElements2);
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testDeviceCopyExecTime() {
        final int numElements = 1000000;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            List<Integer> array = Arrays.asList(new Integer[numElements]);
            long s1 = System.currentTimeMillis();
            for (int i = 0; i < numElements; i++) {
                array.set(i, i);
            }
            long e1 = System.currentTimeMillis();
//            System.out.println("- init java array=" + (e1 - s1) + " ms");

            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements);
            long s2 = System.currentTimeMillis();
            for (int i = 0; i < numElements; ++i) {
                sourceDeviceArray.setArrayElement(i, i);
            }
            long e2 = System.currentTimeMillis();
//            System.out.println("- init grcuda array=" + (e2 - s2) + " ms");

            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                destinationDeviceArray.setArrayElement(i, 0);
            }

            long s3 = System.currentTimeMillis();
            destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements);
            long e3 = System.currentTimeMillis();
//            System.out.println("- grcuda memcpy=" + (e3 - s3) + " ms");

            long s4 = System.currentTimeMillis();
            for (int i = 0; i < numElements; i++) {
                destinationDeviceArray.setArrayElement(i, array.get(i));
            }
            long e4 = System.currentTimeMillis();
//            System.out.println("- java memcpy=" + (e4 - s4) + " ms");

            long s5 = System.currentTimeMillis();
            destinationDeviceArray.invokeMember("copyFrom", array, numElements);
            long e5 = System.currentTimeMillis();
//            System.out.println("- grcuda memcpy - slow path=" + (e5 - s5) + " ms");

            // Verify content of device array
            for (int i = 0; i < numElements; ++i) {
                assertEquals(i, destinationDeviceArray.getArrayElement(i).asInt());
            }
        }
    }

    @Test
    public void testDeviceArrayCopyFromArrayList() {
        final int numElements = 1000000;
        List<Integer> array = Arrays.asList(new Integer[numElements]);
        for (int i = 0; i < numElements; ++i) {
            array.set(i, i + 1);
        }
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            // create DeviceArray and copy content from array list memory into it;
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            Value deviceArray = createDeviceArray.execute("int", numElements);

            long s1 = System.currentTimeMillis();
            deviceArray.invokeMember("copyFrom", array, numElements);
            long e1 = System.currentTimeMillis();
//            System.out.println("- copy from java array=" + (e1 - s1) + " ms");

            // Verify content of device array;
            for (int i = 0; i < numElements; ++i) {
                assertEquals(i + 1, deviceArray.getArrayElement(i).asInt());
            }
        }
    }

    @Test
    public void testDeviceArrayCopyToArrayList() {
        final int numElements = 1000000;
        List<Integer> array = Arrays.asList(new Integer[numElements]);
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            // create DeviceArray and copy content from array list memory into it;
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            Value deviceArray = createDeviceArray.execute("int", numElements);
            long s1 = System.currentTimeMillis();
            for (int i = 0; i < numElements; ++i) {
                deviceArray.setArrayElement(i, i + 1);
            }
            long e1 = System.currentTimeMillis();
//            System.out.println("- init device array=" + (e1 - s1) + " ms");

            long s2 = System.currentTimeMillis();
            deviceArray.invokeMember("copyTo", array, numElements);
            long e2 = System.currentTimeMillis();
//            System.out.println("- copy to device array=" + (e2 - s2) + " ms");

            // Verify content of java array;
            for (int i = 0; i < numElements; ++i) {
                assertEquals(i + 1, array.get(i).intValue());
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyFromArrayList() {
        final int numElements1 = 500;
        final int numElements2 = 2000;
        ArrayList<Integer> array = new ArrayList<>(numElements1 * numElements2);
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {

            // create DeviceArray and copy content from array list memory into it;
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            Value deviceArray = createDeviceArray.execute("int", numElements1, numElements2);

            // Set each value to its index + 1;
            long s1 = System.currentTimeMillis();
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    deviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j + 1);
                }
            }
            long e1 = System.currentTimeMillis();
//            System.out.println("- init 2d device array=" + (e1 - s1) + " ms");

            long s2 = System.currentTimeMillis();
            deviceArray.invokeMember("copyTo", array, numElements1 * numElements2);
            long e2 = System.currentTimeMillis();
//            System.out.println("- copy to 2d device array=" + (e2 - s2) + " ms");

            // Verify content of java array;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    array.add(i * numElements2 + j + 1);
                    assertEquals(i * numElements2 + j + 1, array.get(i * numElements2 + j).intValue());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyToArrayList() {
        final int numElements1 = 500;
        final int numElements2 = 2000;
        ArrayList<Integer> array = new ArrayList<>(numElements1 * numElements2);
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {

            // Set each value to its index + 1;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    array.add(i * numElements2 + j + 1);
                }
            }

            // create DeviceArray and copy content from array list memory into it;
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            Value deviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            long s1 = System.currentTimeMillis();
            deviceArray.invokeMember("copyFrom", array, numElements1 * numElements2);
            long e1 = System.currentTimeMillis();
//            System.out.println("- copy to device array=" + (e1 - s1) + " ms");

            // Verify content of device array;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    array.add(i * numElements2 + j + 1);
                    assertEquals(i * numElements2 + j + 1, deviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }
}
