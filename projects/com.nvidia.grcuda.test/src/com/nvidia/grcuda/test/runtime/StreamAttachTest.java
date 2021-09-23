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

public class StreamAttachTest {

    /**
     * Simply check if we can attach an array to a CUDA stream without blowing things up!
     */
    @Test
    public void attachStreamSimpleTest() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value stream = createStream.execute();

            final int numElements = 100;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);

            Value streamAttach = context.eval("grcuda", "cudaStreamAttachMemAsync");
            assertNotNull(streamAttach);
            assertTrue(streamAttach.canExecute());
            streamAttach.execute(stream, x);

            // Synchronize and destroy the stream;
            Value streamSync = context.eval("grcuda", "cudaStreamSynchronize");
            Value streamDestroy = context.eval("grcuda", "cudaStreamDestroy");
            streamSync.execute(stream);
            streamDestroy.execute(stream);
        }
    }

    /**
     * Check that we can attach many different streams on different arrays;
     */
    @Test
    public void attachManyStreamsTest() {
        int numStreams = 8;
        Set<Value> streamSet = new HashSet<>();
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {

            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value streamAttach = context.eval("grcuda", "cudaStreamAttachMemAsync");
            Value streamSync = context.eval("grcuda", "cudaStreamSynchronize");
            Value streamDestroy = context.eval("grcuda", "cudaStreamDestroy");
            final int numElements = 100;

            IntStream.range(0, numStreams).forEach(i -> {
                Value x = deviceArrayConstructor.execute("float", numElements);
                Value stream = createStream.execute();
                streamAttach.execute(stream, x);
                streamSet.add(stream);
            });
            // Sync and destroy;
            streamSet.forEach(s -> {
                streamSync.execute(s);
                streamDestroy.execute(s);
            });
        }
    }

    /**
     * Check that we can attach multiple arrays to the same stream;
     */
    @Test
    public void attachManyArraysToStreamTest() {
        int numArrays = 4;

        try (Context context = GrCUDATestUtil.buildTestContext().build()) {

            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value streamAttach = context.eval("grcuda", "cudaStreamAttachMemAsync");
            Value streamSync = context.eval("grcuda", "cudaStreamSynchronize");
            Value streamDestroy = context.eval("grcuda", "cudaStreamDestroy");
            final int numElements = 100;

            Value stream = createStream.execute();

            IntStream.range(0, numArrays).forEach(i -> {
                Value x = deviceArrayConstructor.execute("float", numElements);
                streamAttach.execute(stream, x);
            });
            // Sync and destroy;
            streamSync.execute(stream);
            streamDestroy.execute(stream);
        }
    }

    /**
     * Check that we can attach the same array to multiple streams, in sequence;
     */
    @Test
    public void attachManyStreamsToArrayTest() {
        int numStreams = 4;
        Set<Value> streamSet = new HashSet<>();

        try (Context context = GrCUDATestUtil.buildTestContext().build()) {

            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value streamAttach = context.eval("grcuda", "cudaStreamAttachMemAsync");
            Value streamSync = context.eval("grcuda", "cudaStreamSynchronize");
            Value streamDestroy = context.eval("grcuda", "cudaStreamDestroy");
            final int numElements = 100;
            Value x = deviceArrayConstructor.execute("float", numElements);


            IntStream.range(0, numStreams).forEach(i -> {
                Value stream = createStream.execute();
                streamAttach.execute(stream, x);
                streamSync.execute(stream);
                streamSet.add(stream);
            });
            // Sync and destroy;
            streamSet.forEach(streamDestroy::execute);
        }
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
     * Execute a simple kernel on a non-default stream with attached memory;
     */
    @Test
    public void useAttachedStreamTest() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value streamAttach = context.eval("grcuda", "cudaStreamAttachMemAsync");
            Value streamSync = context.eval("grcuda", "cudaStreamSynchronize");
            Value streamDestroy = context.eval("grcuda", "cudaStreamDestroy");

            Value stream = createStream.execute();

            final int numElements = 100;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value x = deviceArrayConstructor.execute("float", numElements);

            // Attach the array to the stream;
            streamAttach.execute(stream, x);
            streamSync.execute(stream);

            // Build the kernel;
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }
            // Execute the kernel on the custom stream;
            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream);
            configuredSquareKernel.execute(x, numElements);
            streamSync.execute(stream);

            for (int i = 0; i < numElements; i++) {
                assertEquals(4.0, x.getArrayElement(i).asFloat(), 0.01);
            }

            streamDestroy.execute(stream);
        }
    }

    /**
     * Execute two simple kernel on non-default streams with attached memory. Array reads synchronize only a single stream,
     * which would cause errors if executed on non-attached memory in pre-Pascal GPUs;
     */
    @Test
    public void useTwoStreamsTest() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value streamAttach = context.eval("grcuda", "cudaStreamAttachMemAsync");
            Value streamSync = context.eval("grcuda", "cudaStreamSynchronize");
            Value streamDestroy = context.eval("grcuda", "cudaStreamDestroy");

            Value stream1 = createStream.execute();
            Value stream2 = createStream.execute();

            final int numElements = 100;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);

            // Attach the array to the stream;
            streamAttach.execute(stream1, x);
            streamSync.execute(stream1);
            streamAttach.execute(stream2, y);
            streamSync.execute(stream2);

            // Build the kernel;
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
                y.setArrayElement(i, 4.0);
            }
            // Execute the kernel on the custom stream;
            Value configuredSquareKernel1 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream1);
            configuredSquareKernel1.execute(x, numElements);
            Value configuredSquareKernel2 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream2);
            configuredSquareKernel2.execute(y, numElements);

            // Sync just one stream before accessing the dataM
            Value syncStream = context.eval("grcuda", "cudaStreamSynchronize");
            syncStream.execute(stream1);
            assertEquals(4.0, x.getArrayElement(0).asFloat(), 0.01);
            syncStream.execute(stream2);
            assertEquals(16.0, y.getArrayElement(0).asFloat(), 0.01);

            // Check the other values;
            for (int i = 1; i < numElements; i++) {
                assertEquals(4.0, x.getArrayElement(i).asFloat(), 0.01);
                assertEquals(16.0, y.getArrayElement(i).asFloat(), 0.01);
            }

            streamDestroy.execute(stream1);
            streamDestroy.execute(stream2);
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

    /**
     * Execute a simple kernel on a non-default stream with attached memory,
     * then move back the memory to the global stream and execute another kernel;
     */
    @Test
    public void useDefaultAttachedStreamTest() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value createStream = context.eval("grcuda", "cudaStreamCreate");
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value streamAttach = context.eval("grcuda", "cudaStreamAttachMemAsync");
            Value streamSync = context.eval("grcuda", "cudaStreamSynchronize");
            Value streamDestroy = context.eval("grcuda", "cudaStreamDestroy");

            Value stream = createStream.execute();

            final int numElements = 100;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value x = deviceArrayConstructor.execute("float", numElements);

            // Attach the array to the stream;
            streamAttach.execute(stream, x);
            streamSync.execute(stream);

            // Build the kernel;
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }
            // Execute the kernel on the custom stream;
            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK, stream);
            configuredSquareKernel.execute(x, numElements);
            streamSync.execute(stream);

            for (int i = 0; i < numElements; i++) {
                assertEquals(4.0, x.getArrayElement(i).asFloat(), 0.01);
            }

            // Reset the visibility of the array;
            streamAttach.execute(stream, x, 0x01);
            configuredSquareKernel.execute(x, numElements);

            streamSync.execute(stream);

            for (int i = 0; i < numElements; i++) {
                assertEquals(16.0, x.getArrayElement(i).asFloat(), 0.01);
            }

            // Reset the array to use the default stream, by providing just the array;
            streamAttach.execute(x);
            Value configuredSquareKernel2 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            configuredSquareKernel2.execute(x, numElements);

            Value deviceSync = context.eval("grcuda", "cudaDeviceSynchronize");
            deviceSync.execute();

            for (int i = 0; i < numElements; i++) {
                assertEquals(256, x.getArrayElement(i).asFloat(), 0.01);
            }

            streamDestroy.execute(stream);
        }
    }
}
