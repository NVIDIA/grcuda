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
package com.nvidia.grcuda.test;

import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class CUDAEventTest {

    /**
     * Simply check if we can create a CUDA event without blowing things up!
     */
    @Test
    public void createEventSimpleTest() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createEvent = context.eval("grcuda", "cudaEventCreate");
            Value event = createEvent.execute();
            assertNotNull(event);
            assertTrue(event.isNativePointer());
        }
    }

    /**
     * Check that we can create many different events;
     */
    @Test
    public void createManyEventsTest() {
        int numEvents = 8;
        Set<Long> eventSet = new HashSet<>();
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createEvent = context.eval("grcuda", "cudaEventCreate");
            IntStream.range(0, numEvents).forEach(i -> {
                Value event = createEvent.execute();
                eventSet.add(event.asNativePointer());
                assertNotNull(event);
                assertTrue(event.isNativePointer());
            });
            assertEquals(numEvents, eventSet.size());
        }
    }

    @Test
    public void eventDestroyTest() {
        int numEvents = 8;
        Set<Value> eventSet = new HashSet<>();
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createEvent = context.eval("grcuda", "cudaEventCreate");
            Value destroyEvent = context.eval("grcuda", "cudaEventDestroy");
            IntStream.range(0, numEvents).forEach(i -> {
                Value event = createEvent.execute();
                eventSet.add(event);
                assertNotNull(event);
                assertTrue(event.isNativePointer());
            });
            assertEquals(numEvents, eventSet.size());
            eventSet.forEach(destroyEvent::execute);
        }
    }

    private static final int NUM_THREADS_PER_BLOCK = 32;

    private static final String SQUARE_KERNEL =
            "extern \"C\" __global__ void square(float* x, float *y, int n) {\n" +
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (idx < n) {\n" +
                    "       y[idx] = x[idx] * x[idx];\n" +
                    "    }" +
                    "}\n";

    private static final String SUM_KERNEL =
            "extern \"C\" __global__ void square(float* x, float* y, int n) {\n" +
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (idx < n) {\n" +
                    "       x[idx] = x[idx] + y[idx];\n" +
                    "    }" +
                    "}\n";

    /**
     * Execute sequentially two simple kernel on non-default streams, and synchronize the execution using events;
     */
    @Test
    public void syncStreamsTest() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value stream1 = createStream.execute();
            Value stream2 = createStream.execute();

            final int numElements = 100;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, pointer, sint32");
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }
            // Set the custom streams;
            Value configuredSquareKernel1 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream1);
            Value configuredSquareKernel2 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream2);

            Value createEvent = context.eval("grcuda", "cudaEventCreate");
            Value eventRecord = context.eval("grcuda", "cudaEventRecord");
            Value streamEventWait = context.eval("grcuda", "cudaStreamWaitEvent");

            configuredSquareKernel1.execute(x, y, numElements);

            // Create an event to ensure that kernel 2 executes after kernel 1 is completed;
            Value event = createEvent.execute();
            eventRecord.execute(event, stream1);
            streamEventWait.execute(stream2, event);

            configuredSquareKernel2.execute(y, x, numElements);

            Value syncStream = context.eval("grcuda", "cudaStreamSynchronize");
            syncStream.execute(stream2);

            for (int i = 0; i < numElements; i++) {
                assertEquals(16.0, x.getArrayElement(i).asFloat(), 0.01);
                assertEquals(4.0, y.getArrayElement(i).asFloat(), 0.01);
            }
        }
    }

    /**
     * Execute two kernels on non-default streams, and synchronize them with events before running a third kernel;
     * K1(Y = X^2) -> K3(Y += Z)
     * K2(Z = X^2) /
     */
    @Test
    public void joinComputationsTest() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value stream1 = createStream.execute();
            Value stream2 = createStream.execute();

            final int numElements = 100;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, pointer, sint32");
            Value sumKernel = buildkernel.execute(SUM_KERNEL, "square", "pointer, pointer, sint32");

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }
            // Set the custom streams;
            Value configuredSquareKernel1 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream1);
            Value configuredSquareKernel2 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream2);
            Value configuredSquareKernel3 = sumKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream1);

            Value createEvent = context.eval("grcuda", "cudaEventCreate");
            Value eventRecord = context.eval("grcuda", "cudaEventRecord");
            Value streamEventWait = context.eval("grcuda", "cudaStreamWaitEvent");

            configuredSquareKernel1.execute(x, y, numElements);
            configuredSquareKernel2.execute(x, z, numElements);

            // Create an event to ensure that kernel 2 executes after kernel 1 is completed;
            Value event = createEvent.execute();
            eventRecord.execute(event, stream2);
            streamEventWait.execute(stream1, event);

            configuredSquareKernel3.execute(y, z, numElements);

            Value syncStream = context.eval("grcuda", "cudaStreamSynchronize");
            syncStream.execute(stream1);

            for (int i = 0; i < numElements; i++) {
                assertEquals(8, y.getArrayElement(i).asFloat(), 0.01);
            }
        }
    }
}
