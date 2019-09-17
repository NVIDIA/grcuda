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

import java.util.Random;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;

public class BuildKernelTest {

    /** CUDA C++ source code of incrementing kernel. */
    private static final String INCREMENT_KERNEL_SOURCE = "template <typename T>                     \n" +
                    "__global__ void inc_kernel(T *out_arr, const T *in_arr, size_t num_elements) {  \n" +
                    "  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;    \n" +
                    "       idx += gridDim.x * blockDim.x) {                                         \n" +
                    "    out_arr[idx] = in_arr[idx] + (T{} + 1);                                     \n" +
                    "  }                                                                             \n" +
                    "}\n";

    /** NFI Signature of incrementing kernel. */
    private static final String INCREMENT_KERNEL_SIGNATURE = "pointer, pointer, uint64";

    @Test
    public void testBuildKernel() {
        // See if inc_kernel can be built
        try (Context context = Context.newBuilder().allowAllAccess(true).build()) {
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value incrKernel = buildkernel.execute(INCREMENT_KERNEL_SOURCE, "inc_kernel<int>", INCREMENT_KERNEL_SIGNATURE);
            assertNotNull(incrKernel);
            assertTrue(incrKernel.canExecute());
            assertEquals(0, incrKernel.getMember("launchCount").asInt());
            assertNotNull(incrKernel.getMember("ptx").asString());
        }
    }

    @Test
    public void testBuild1DKernelAndLaunch() {
        // Build inc_kernel, launch it, and check results.
        try (Context context = Context.newBuilder().allowAllAccess(true).build()) {
            final int numElements = 1000;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value incrKernel = buildkernel.execute(INCREMENT_KERNEL_SOURCE, "inc_kernel<int>", INCREMENT_KERNEL_SIGNATURE);
            assertNotNull(incrKernel);
            assertTrue(incrKernel.canExecute());
            assertEquals(0, incrKernel.getMember("launchCount").asInt());
            assertNotNull(incrKernel.getMember("ptx").asString());
            Value inDevArray = deviceArrayConstructor.execute("int", numElements);
            Value outDevArray = deviceArrayConstructor.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                inDevArray.setArrayElement(i, i);
                outDevArray.setArrayElement(i, 0);
            }
            // Execute kernel as <<<8, 128>>> that is 8 blocks with 128 threads each
            Value configuredIncKernel = incrKernel.execute(8, 128);
            assertTrue(configuredIncKernel.canExecute());
            configuredIncKernel.execute(outDevArray, inDevArray, numElements);
            // implicit sync

            // verify results
            for (int i = 0; i < numElements; ++i) {
                assertEquals(i, inDevArray.getArrayElement(i).asInt());
                assertEquals(i + 1, outDevArray.getArrayElement(i).asInt());
            }
            assertEquals(1, incrKernel.getMember("launchCount").asInt());
        }
    }

    /** CUDA C source code simple matrix-multiplication kernel. */
    private static final String MATMULT_KERNEL_SOURCE = "\n" +
                    "__global__ void matmult(int num_a_rows, int num_a_cols, int num_b_cols,\n" +
                    "                        int block_size, const float * __restrict matrix_a,\n" +
                    "                        const float * __restrict matrix_b,\n" +
                    "                        float * __restrict matrix_c) {\n" +
                    "  int block_row = blockIdx.y;\n" +
                    "  int block_col = blockIdx.x;\n" +
                    "  int row = threadIdx.y;\n" +
                    "  int col = threadIdx.x;\n" +
                    "  extern __shared__ float shmem[];\n" +
                    "\n" +
                    "  // sub matrix A\n" +
                    "  int sub_a_begin = num_a_cols * block_size * block_row;\n" +
                    "  int sub_a_end = sub_a_begin + num_a_cols - 1;\n" +
                    "  int sub_a_step_size = block_size;\n" +
                    "  float *sub_a_matrix = &shmem[0];\n" +
                    "\n" +
                    "  // sub matrix B\n" +
                    "  int sub_b_begin = block_size * block_col;\n" +
                    "  int sub_b_step_size = block_size * num_b_cols;\n" +
                    "  float *sub_b_matrix = &shmem[block_size * block_size];\n" +
                    "\n" +
                    "  float sub_c_val = 0;\n" +
                    "\n" +
                    "  // iterate over all tiles\n" +
                    "  for (int a = sub_a_begin, b = sub_b_begin; a <= sub_a_end;\n" +
                    "       a += sub_a_step_size, b += sub_b_step_size) {\n" +
                    "    // bring A and B tiles into shared memory\n" +
                    "    sub_a_matrix[row * block_size + col] = matrix_a[a + num_a_cols * row + col];\n" +
                    "    sub_b_matrix[row * block_size + col] = matrix_b[b + num_b_cols * row + col];\n" +
                    "    __syncthreads();\n" +
                    "\n" +
                    "    // matrix-multiply tiles from shared memory\n" +
                    "    for (int k = 0; k < block_size; ++k) {\n" +
                    "      sub_c_val += sub_a_matrix[row * block_size + k] *\n" +
                    "                   sub_b_matrix[k * block_size + col];\n" +
                    "    }\n" +
                    "    __syncthreads();\n" +
                    "  }\n" +
                    "\n" +
                    "  // write back result element\n" +
                    "  matrix_c[num_b_cols * block_size * block_row + block_size * block_col +\n" +
                    "           num_b_cols * row + col] = sub_c_val;\n" +
                    "}\n";

    /** NFI Signature of matrix-multiplication kernel. */
    private static final String MATMULT_KERNEL_SIGNATURE = "sint32, sint32, sint32, sint32, pointer, pointer, pointer";

    @Test
    public void testBuild2DKernelAndLaunch() {
        // build matmult kernel, launch it on 2D grid, and check results
        try (Context context = Context.newBuilder().allowAllAccess(true).build()) {
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value matmultKernel = buildkernel.execute(MATMULT_KERNEL_SOURCE, "matmult", MATMULT_KERNEL_SIGNATURE);
            assertNotNull(matmultKernel);
            assertTrue(matmultKernel.canExecute());
            assertEquals(0, matmultKernel.getMember("launchCount").asInt());
            assertNotNull(matmultKernel.getMember("ptx").asString());

            // generate matrices
            final int numARows = 256;
            final int numACols = 192;
            final int numBRows = numACols;
            final int numBCols = 128;
            final int blockSize = 32;
            Value matrixA = context.eval("grcuda", "float[" + (numARows * numACols) + "]");
            Value matrixB = context.eval("grcuda", "float[" + (numBRows * numBCols) + "]");
            Value matrixC = context.eval("grcuda", "float[" + (numARows * numBCols) + "]");
            fillRandomMatrix(numARows, numACols, matrixA);
            fillRandomMatrix(numBRows, numBCols, matrixB);

            // launch kernel
            int[] threads = {blockSize, blockSize};
            int[] blocks = {numBCols / blockSize, numARows / blockSize};
            int shmemBytes = 2 * 4 * blockSize * blockSize;
            matmultKernel.execute(blocks, threads, shmemBytes).execute(numARows, numACols,
                            numBCols, blockSize, matrixA, matrixB, matrixC);
            // implicit synchronization
            assertEquals(1, matmultKernel.getMember("launchCount").asInt());

            // verify results with vanilla matrix multiplication
            for (int i = 0; i < numARows; i++) {
                for (int j = 0; j < numBCols; j++) {
                    float t = 0;
                    for (int k = 0; k < numACols; k++) {
                        t += matrixA.getArrayElement(i * numACols + k).asFloat() * matrixB.getArrayElement(k * numBCols + j).asFloat();
                    }
                    float c = matrixC.getArrayElement(i * numBCols + j).asFloat();
                    assertEquals(t, c, 1e-3);
                }
            }
        }
    }

    private static void fillRandomMatrix(int numRows, int numCols, Value matrix) {
        Random rand = new Random(42);
        for (int i = 0; i < numRows * numCols; i++) {
            matrix.setArrayElement(i, (float) rand.nextGaussian());
        }
    }
}
