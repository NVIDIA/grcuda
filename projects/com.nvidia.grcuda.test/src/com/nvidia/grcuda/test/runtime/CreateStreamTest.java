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
package com.nvidia.grcuda.test.runtime;

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

public class CreateStreamTest {

    /**
     * Simply check if we can create a CUDA stream without blowing things up!
     */
    @Test
    public void createStreamSimpleTest() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value stream = createStream.execute();
            assertNotNull(stream);
            assertTrue(stream.isNativePointer());
        }
    }

    /**
     * Check that we can create many different streams;
     */
    @Test
    public void createManyStreamsTest() {
        int numStreams = 8;
        Set<Long> streamSet = new HashSet<>();
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            IntStream.range(0, numStreams).forEach(i -> {
                Value createStream = context.eval("grcuda", "cudaStreamCreate");
                Value stream = createStream.execute();
                streamSet.add(stream.asNativePointer());
                assertNotNull(stream);
                assertTrue(stream.isNativePointer());
            });
        }
        assertEquals(numStreams, streamSet.size());
    }

    private static final int NUM_THREADS_PER_BLOCK = 32;

    private static final String SQUARE_KERNEL =
            "extern \"C\" __global__ void square(float* x, int n) {\n" +
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (idx < n) {\n" +
                    "       x[idx] = x[idx] * x[idx];\n" +
                    "    }" +
                    "}\n";

    /**
     * Execute a simple kernel on a non-default stream;
     */
    @Test
    public void useStreamTest() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value stream = createStream.execute();
            assertNotNull(stream);
            assertTrue(stream.isNativePointer());

            final int numElements = 100;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }
            // Set the custom stream;
            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream);
            configuredSquareKernel.execute(x, numElements);

            // Wait for the computations to end;
            Value syncStream = context.eval("grcuda", "cudaDeviceSynchronize");
            syncStream.execute();

            for (int i = 0; i < numElements; i++) {
                assertEquals(4.0, x.getArrayElement(i).asFloat(), 0.01);
            }
        }
    }

    /**
     * Execute two simple kernel on non-default streams;
     */
    @Test
    public void useTwoStreamsTest() {
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
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
                y.setArrayElement(i, 4.0);
            }
            // Set the custom streams;
            Value configuredSquareKernel1 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream1);
            Value configuredSquareKernel2 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream2);

            configuredSquareKernel1.execute(x, numElements);
            configuredSquareKernel2.execute(y, numElements);

            // Wait for the computations to end;
            Value syncStream = context.eval("grcuda", "cudaDeviceSynchronize");
            syncStream.execute();

            for (int i = 0; i < numElements; i++) {
                assertEquals(4.0, x.getArrayElement(i).asFloat(), 0.01);
                assertEquals(16.0, y.getArrayElement(i).asFloat(), 0.01);
            }
        }
    }

    /**
     * Execute two simple kernel on non-default streams, and synchronize each stream independently;
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
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
                y.setArrayElement(i, 4.0);
            }
            // Set the custom streams;
            Value configuredSquareKernel1 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream1);
            Value configuredSquareKernel2 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream2);

            configuredSquareKernel1.execute(x, numElements);
            configuredSquareKernel2.execute(y, numElements);

            Value syncStream = context.eval("grcuda", "cudaStreamSynchronize");
            syncStream.execute(stream1);
            syncStream.execute(stream2);

            for (int i = 0; i < numElements; i++) {
                assertEquals(4.0, x.getArrayElement(i).asFloat(), 0.01);
                assertEquals(16.0, y.getArrayElement(i).asFloat(), 0.01);
            }
        }
    }


    @Test
    public void streamDestroyTest() {
        int numStreams = 8;
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Set<Value> streamSet = new HashSet<>();
            IntStream.range(0, numStreams).forEach(i -> {
                Value createStream = context.eval("grcuda", "cudaStreamCreate");
                Value stream = createStream.execute();
                streamSet.add(stream);
                assertNotNull(stream);
            });
            Value destroyStream = context.eval("grcuda", "cudaStreamDestroy");
            streamSet.forEach(destroyStream::execute);
        }
    }
}
