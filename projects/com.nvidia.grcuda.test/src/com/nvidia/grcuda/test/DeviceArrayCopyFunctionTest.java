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

import com.nvidia.grcuda.ComputationArgumentWithValue;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.array.DeviceArray;
import com.nvidia.grcuda.array.MultiDimDeviceArray;
import com.nvidia.grcuda.array.MultiDimDeviceArrayView;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.nvidia.grcuda.gpu.computation.ArrayCopyFunctionExecutionInitializer;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.test.mock.GrCUDAExecutionContextMock;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;
import com.nvidia.grcuda.gpu.LittleEndianNativeArrayView;
import com.nvidia.grcuda.gpu.OffheapMemory;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

@RunWith(Enclosed.class)
public class DeviceArrayCopyFunctionTest {

    public static class DeviceArrayCopyFunctionTestNotParameterized {
        protected static class DeviceArrayMock extends DeviceArray {
            DeviceArrayMock() {
                super(new GrCUDAExecutionContextMock(), 0, Type.SINT32);
            }

            @Override
            protected LittleEndianNativeArrayView allocateMemory() {
                return null;
            }
        }

        protected static class MultiDimDeviceArrayMock extends MultiDimDeviceArray {
            MultiDimDeviceArrayMock(long[] dimensions, boolean columnMajor) {
                super(new GrCUDAExecutionContextMock(), Type.SINT32, dimensions, columnMajor);
            }

            @Override
            protected LittleEndianNativeArrayView allocateMemory() {
                return null;
            }
        }

        @Test
        public void testIfSlowPathIsChosenCorrectly() {
            try (Context ctx = Context.newBuilder().allowAllAccess(true).allowExperimentalOptions(true).logHandler(new TestLogHandler())
                    .option("log.grcuda.com.nvidia.grcuda.level", "SEVERE").build()) {
                ctx.getEngine(); // ctx is required to set the logger level. Perform a access to suppress compiler warnings about it being unused;
                DeviceArray array1d = new DeviceArrayMock();
                DeviceArray array1d2 = new DeviceArrayMock();
                DeviceArrayCopyFunction copyFunction = new DeviceArrayCopyFunction(array1d, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
                assertTrue(copyFunction.canUseMemcpy(array1d2));

                long[] dimensions = {2, 2};
                MultiDimDeviceArray array2d = new MultiDimDeviceArrayMock(dimensions, false);
                MultiDimDeviceArray array2d2 = new MultiDimDeviceArrayMock(dimensions, false);
                copyFunction = new DeviceArrayCopyFunction(array2d, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
                assertTrue(copyFunction.canUseMemcpy(array2d2));

                // Inconsistent memory layouts;
                MultiDimDeviceArray array2d3 = new MultiDimDeviceArrayMock(dimensions, true);
                copyFunction = new DeviceArrayCopyFunction(array2d, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
                assertFalse(copyFunction.canUseMemcpy(array2d3));

                // We can copy from a single row, if the layout is row-major;
                MultiDimDeviceArrayView view1 = new MultiDimDeviceArrayView(array2d2, 1, 0, 0);
                copyFunction = new DeviceArrayCopyFunction(view1, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
                assertTrue(copyFunction.canUseMemcpy(array2d));
                copyFunction = new DeviceArrayCopyFunction(array1d, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
                assertTrue(copyFunction.canUseMemcpy(array2d));

                // We cannot copy from a single row, if the destination layout is column-major;
                MultiDimDeviceArrayView view2 = new MultiDimDeviceArrayView(array2d3, 1, 0, 0);
                copyFunction = new DeviceArrayCopyFunction(view2, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
                assertFalse(copyFunction.canUseMemcpy(array2d));
                copyFunction = new DeviceArrayCopyFunction(view2, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
                assertFalse(copyFunction.canUseMemcpy(array2d3));
                copyFunction = new DeviceArrayCopyFunction(array1d, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
                assertFalse(copyFunction.canUseMemcpy(array2d3));
            }
        }

        @Test
        public void testDeviceToDeviceDependencies() {
            try (Context ctx = Context.newBuilder().allowAllAccess(true).allowExperimentalOptions(true).logHandler(new TestLogHandler())
                    .option("log.grcuda.com.nvidia.grcuda.level", "SEVERE").build()) {
                ctx.getEngine(); // ctx is required to set the logger level. Perform a access to suppress compiler warnings about it being unused;
                DeviceArray array1d = new DeviceArrayMock();
                DeviceArray array1d2 = new DeviceArrayMock();
                ArrayCopyFunctionExecutionInitializer init = new ArrayCopyFunctionExecutionInitializer(array1d, array1d2, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
                List<ComputationArgumentWithValue> deps = init.initialize();
                assertEquals(2, deps.size());
                assertEquals(array1d, deps.get(0).getArgumentValue());
                assertEquals(array1d2, deps.get(1).getArgumentValue());
                assertFalse(deps.get(0).isConst());
                assertTrue(deps.get(1).isConst());

                init = new ArrayCopyFunctionExecutionInitializer(array1d, array1d2, DeviceArrayCopyFunction.CopyDirection.TO_POINTER);
                deps = init.initialize();
                assertEquals(2, deps.size());
                assertEquals(array1d, deps.get(0).getArgumentValue());
                assertEquals(array1d2, deps.get(1).getArgumentValue());
                assertTrue(deps.get(0).isConst());
                assertFalse(deps.get(1).isConst());

                int[] array = {1, 2, 3};
                init = new ArrayCopyFunctionExecutionInitializer(array1d, array, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
                deps = init.initialize();
                assertEquals(1, deps.size());
                assertEquals(array1d, deps.get(0).getArgumentValue());
                assertFalse(deps.get(0).isConst());

                int[] array2 = {1, 2, 3};
                init = new ArrayCopyFunctionExecutionInitializer(array1d, array2, DeviceArrayCopyFunction.CopyDirection.TO_POINTER);
                deps = init.initialize();
                assertEquals(1, deps.size());
                assertEquals(array1d, deps.get(0).getArgumentValue());
                assertTrue(deps.get(0).isConst());
            }
        }

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
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }

                // create destination device array.
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements2);
                // Set each value to j;
                for (int j = 0; j < numElements2; ++j) {
                    destinationDeviceArray.setArrayElement(j, 42 + j);
                }

                sourceDeviceArray.getArrayElement(3).invokeMember("copyFrom", destinationDeviceArray, sourceDeviceArray.getArrayElement(3).getArraySize());
                // Verify content of device array
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(42 + j, sourceDeviceArray.getArrayElement(3).getArrayElement(j).asInt());
                }
                // Everything else is unmodified;
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        if (i != 3) {
                            assertEquals(i * numElements2 + j, sourceDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                        }
                    }
                }
            }
        }

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
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }
                destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements1 * numElements2);
                // Verify content of device array
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        assertEquals(i * numElements2 + j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
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
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }
                // create destination device array initialize its elements to zero.
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");

                sourceDeviceArray.invokeMember("copyTo", destinationDeviceArray, numElements1 * numElements2);
                // Verify content of device array
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        assertEquals(i * numElements2 + j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                    }
                }
            }
        }

        @Test
        public void testMultiDimDeviceArrayCopyFromDeviceArrayC() {
            final int numElements1 = 10;
            final int numElements2 = 25;
            try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
                Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
                // create device array initialize its elements.
                Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "C");

                // create destination device array initialize its elements to zero.
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "C");

                // Set each row to j;
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }
                destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements1 * numElements2);
                // Verify content of device array
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        assertEquals(i * numElements2 + j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                    }
                }
            }
        }

        @Test
        public void testMultiDimDeviceArrayCopyToDeviceArrayC() {
            final int numElements1 = 10;
            final int numElements2 = 25;
            try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
                Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
                // create device array initialize its elements.
                Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "C");
                // Set each row to j;
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, j);
                    }
                }
                // create destination device array initialize its elements to zero.
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "C");

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
        public void testMultiDimDeviceArrayCopyToDeviceArrayRowC() {
            final int numElements1 = 5;
            final int numElements2 = 7;
            try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
                Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
                // Create device array initialize its elements;
                Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
                // Initialize elements with unique values.
                // Values are still written as (row, col), even if the storage is "C";
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }
                // Initialize destination array with unique values, to ensure that it's modified correctly;
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        destinationDeviceArray.getArrayElement(i).setArrayElement(j, -(i * numElements2 + j));
                    }
                }

                // This copies the 4th row of the source array into the 4th row of the destination array;
                sourceDeviceArray.getArrayElement(3).invokeMember("copyTo", destinationDeviceArray.getArrayElement(3), sourceDeviceArray.getArrayElement(3).getArraySize());

                // Verify content of device array
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        assertEquals((i == 3 ? 1 : -1) * (i * numElements2 + j), destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                    }
                }
            }
        }

        @Test
        public void testMultiDimDeviceArrayCopyFromDeviceArrayRowC() {
            final int numElements1 = 10;
            final int numElements2 = 25;
            try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
                Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
                // Create device array initialize its elements;
                Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
                // Initialize elements with unique values.
                // Values are still written as (row, col), even if the storage is "C";
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }
                // Initialize destination array with unique values, to ensure that it's modified correctly;
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        destinationDeviceArray.getArrayElement(i).setArrayElement(j, -(i * numElements2 + j));
                    }
                }

                sourceDeviceArray.getArrayElement(3).invokeMember("copyFrom", destinationDeviceArray.getArrayElement(3), destinationDeviceArray.getArrayElement(3).getArraySize());
                // Verify content of device array
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        assertEquals((i == 3 ? -1 : 1) * (i * numElements2 + j), sourceDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                    }
                }
            }
        }

        @Test
        public void testMultiDimDeviceArrayCopyToDeviceArrayRowF() {
            final int numElements1 = 5;
            final int numElements2 = 7;
            try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
                Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
                // Create device array initialize its elements;
                Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");
                // Initialize elements with unique values.
                // Values are still written as (row, col), even if the storage is "C";
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }
                // Initialize destination array with unique values, to ensure that it's modified correctly;
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        destinationDeviceArray.getArrayElement(i).setArrayElement(j, -(i * numElements2 + j));
                    }
                }

                // This copies the 4th column of the source array into the 4th column of the destination array;
                sourceDeviceArray.getArrayElement(3).invokeMember("copyTo", destinationDeviceArray.getArrayElement(3), sourceDeviceArray.getArrayElement(3).getArraySize());

                // Verify content of device array
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        assertEquals((i == 3 ? 1 : -1) * (i * numElements2 + j), destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                    }
                }
            }
        }

        @Test
        public void testMultiDimDeviceArrayCopyFromDeviceArrayRowF() {
            final int numElements1 = 5;
            final int numElements2 = 7;
            try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
                Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
                // Create device array initialize its elements;
                Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");
                // Initialize elements with unique values.
                // Values are still written as (row, col), even if the storage is "C";
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }
                // Initialize destination array with unique values, to ensure that it's modified correctly;
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        destinationDeviceArray.getArrayElement(i).setArrayElement(j, -(i * numElements2 + j));
                    }
                }

                // This copies the 4th column of the source array into the 4th column of the destination array;
                sourceDeviceArray.getArrayElement(3).invokeMember("copyFrom", destinationDeviceArray.getArrayElement(3), sourceDeviceArray.getArrayElement(3).getArraySize());

                // Verify content of device array
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        assertEquals((i == 3 ? -1 : 1) * (i * numElements2 + j), sourceDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                    }
                }
            }
        }
    }

    @RunWith(Parameterized.class)
    public static class DeviceArrayCopyFunctionParametrized {
        @Parameterized.Parameters
        public static Collection<Object[]> data() {
            return GrCUDATestUtil.getAllOptionCombinations();
        }

        private final GrCUDATestOptionsStruct options;

        public DeviceArrayCopyFunctionParametrized(GrCUDATestOptionsStruct options) {
            this.options = options;
        }


        private static final int NUM_THREADS_PER_BLOCK = 32;

        private static final String ADD_ONE_KERNEL =
                "extern \"C\" __global__ void add(int* x, int n) {\n" +
                        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                        "    if (idx < n) {\n" +
                        "       x[idx] = x[idx] + 1;\n" +
                        "    }" +
                        "}\n";

        @Test
        public void testDeviceDeviceMemcpyDependency() {
            final int numElements1 = 25;
            final int numElements2 = 50;
            final int numBlocks = (numElements1 * numElements2 + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
                Value createDeviceArray = context.eval("grcuda", "DeviceArray");
                Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);

                Value buildkernel = context.eval("grcuda", "buildkernel");
                Value addKernel = buildkernel.execute(ADD_ONE_KERNEL, "add", "pointer, sint32");

                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }

                // Call kernel;
                addKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK).execute(sourceDeviceArray, numElements1 * numElements2);
                // Copy values from source to destination, using copyFrom;
                destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements1 * numElements2);
                // Verify content of device array
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        assertEquals((i * numElements2 + j) + 1, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                    }
                }
            }
        }

        @Test
        public void testDeviceDeviceMemcpyDependencySingleRow() {
            final int numElements1 = 25;
            final int numElements2 = 50;
            final int numBlocks = (numElements1 * numElements2 + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
                Value createDeviceArray = context.eval("grcuda", "DeviceArray");
                Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);

                Value buildkernel = context.eval("grcuda", "buildkernel");
                Value addKernel = buildkernel.execute(ADD_ONE_KERNEL, "add", "pointer, sint32");

                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }

                // Call kernel;
                addKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK).execute(sourceDeviceArray, numElements1 * numElements2);
                // Copy values from source to destination, using copyFrom;
                destinationDeviceArray.getArrayElement(3).invokeMember("copyFrom", sourceDeviceArray.getArrayElement(3), sourceDeviceArray.getArrayElement(3).getArraySize());
                // Verify content of device array
                for (int i = 0; i < destinationDeviceArray.getArrayElement(3).getArraySize(); ++i) {
                    assertEquals(sourceDeviceArray.getArrayElement(3).getArrayElement(i).asInt(), destinationDeviceArray.getArrayElement(3).getArrayElement(i).asInt());
                }
            }
        }

        @Test
        public void testDeviceDeviceMemcpyDependencyTwoKernelsCopyFrom() {
            final int numElements1 = 25;
            final int numElements2 = 50;
            final int numBlocks = (numElements1 * numElements2 + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
                Value createDeviceArray = context.eval("grcuda", "DeviceArray");
                Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);

                Value buildkernel = context.eval("grcuda", "buildkernel");
                Value addKernel = buildkernel.execute(ADD_ONE_KERNEL, "add", "pointer, sint32");

                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        destinationDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j + 10);
                    }
                }

                // Call kernel;
                addKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK).execute(sourceDeviceArray, numElements1 * numElements2);
                addKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK).execute(destinationDeviceArray, numElements1 * numElements2);
                // Copy values from source to destination, using copyFrom;
                destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements1 * numElements2);
                // Verify content of device array
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        assertEquals((i * numElements2 + j) + 1, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                    }
                }
            }
        }

        @Test
        public void testDeviceDeviceMemcpyDependencyTwoKernelsCopyTo() {
            final int numElements1 = 25;
            final int numElements2 = 50;
            final int numBlocks = (numElements1 * numElements2 + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
                Value createDeviceArray = context.eval("grcuda", "DeviceArray");
                Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
                Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);

                Value buildkernel = context.eval("grcuda", "buildkernel");
                Value addKernel = buildkernel.execute(ADD_ONE_KERNEL, "add", "pointer, sint32");

                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                    }
                }
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        destinationDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j + 10);
                    }
                }

                // Call kernel;
                addKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK).execute(sourceDeviceArray, numElements1 * numElements2);
                addKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK).execute(destinationDeviceArray, numElements1 * numElements2);
                // Copy values from source to destination, using copyFrom;
                sourceDeviceArray.invokeMember("copyTo", destinationDeviceArray, numElements1 * numElements2);
                // Verify content of device array
                for (int i = 0; i < numElements1; ++i) {
                    for (int j = 0; j < numElements2; ++j) {
                        assertEquals((i * numElements2 + j) + 1, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                    }
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
