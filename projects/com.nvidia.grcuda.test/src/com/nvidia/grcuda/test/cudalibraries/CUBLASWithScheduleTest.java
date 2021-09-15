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
package com.nvidia.grcuda.test.cudalibraries;

import static org.junit.Assert.assertEquals;

import java.util.Collection;
import java.util.function.Function;

import com.nvidia.grcuda.test.util.GrCUDATestOptionsStruct;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

@RunWith(Parameterized.class)
public class CUBLASWithScheduleTest {

    private static final int NUM_THREADS_PER_BLOCK = 32;

    private static final String SCALE_KERNEL =
            "extern \"C\" __global__ void scale(double* y, const double* x, double alpha, int n) {\n" +
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (idx < n) {\n" +
                    "       y[idx] = alpha * x[idx];\n" +
                    "    }" +
                    "}\n";

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return GrCUDATestUtil.getAllOptionCombinations();
    }

    private final GrCUDATestOptionsStruct options;
    private final char typeChar = 'D';

    public CUBLASWithScheduleTest(GrCUDATestOptionsStruct options) {
        this.options = options;
    }

    /**
     * Test 2 independent kernels followed by a BLAS kernel
     * A ---> C
     * B --/
     */
    @Test
    public void testTaxpyJoinPattern() {
        // x = (0, 1, 2, ..., numElements-1)
        // y = (0, 2, 4, ..., 2*(numElements-1))
        // z = (0, 0, 0, ..., 0)
        // z := 2 * x
        // y := 2 * y
        // z := -1 * z + y
        // z = 2 * (0, 1, 2, ..., numElements-1)
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            String cudaType = "double";
            int numElements = 1000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value alpha = deviceArrayConstructor.execute(cudaType, 1);
            alpha.setArrayElement(0, -1);

            // Create some arrays;
            Value x = deviceArrayConstructor.execute(cudaType, numElements);
            Value y = deviceArrayConstructor.execute(cudaType, numElements);
            Value z = deviceArrayConstructor.execute(cudaType, numElements);
            assertEquals(numElements, x.getArraySize());
            assertEquals(numElements, y.getArraySize());
            assertEquals(numElements, z.getArraySize());

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, i);
                y.setArrayElement(i, 2 * i);
                z.setArrayElement(i, 0);
            }

            // Define kernels;
            Value taxpy = context.eval("grcuda", "BLAS::cublas" + typeChar + "axpy");
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value scaleKernel = buildkernel.execute(SCALE_KERNEL, "scale", "pointer, const pointer, double, sint32");
            Value configuredScaleKernel = scaleKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredScaleKernel.execute(z, x, 2, numElements);
            configuredScaleKernel.execute(y, y, 2, numElements);
            taxpy.execute(numElements, alpha, y, 1, z, 1);

            assertOutputVectorIsCorrect(numElements, z, (Integer i) -> -2 * i);
        }
    }

    /**
     * Test a BLAS kernel followed by 2 independent kernels;
     * A--->B
     *  \-->C
     */
    @Test
    public void testTaxpyForkPattern() {
        // x = (0, 1, 2, ..., numElements-1)
        // y = (0, 2, 4, ..., 2*(numElements-1))
        // z = (0, 0, 0, ..., 0)
        // x := -1 * x + y
        // x = (0, 1, 2, ..., numElements-1)
        // z := 2 * x
        // y := 2 * y
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            String cudaType = "double";
            int numElements = 1000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value alpha = deviceArrayConstructor.execute(cudaType, 1);
            alpha.setArrayElement(0, -1);

            // Create some arrays;
            Value x = deviceArrayConstructor.execute(cudaType, numElements);
            Value y = deviceArrayConstructor.execute(cudaType, numElements);
            Value z = deviceArrayConstructor.execute(cudaType, numElements);
            assertEquals(numElements, x.getArraySize());
            assertEquals(numElements, y.getArraySize());
            assertEquals(numElements, z.getArraySize());

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, i);
                y.setArrayElement(i, 2 * i);
                z.setArrayElement(i, 0);
            }

            // Define kernels;
            Value taxpy = context.eval("grcuda", "BLAS::cublas" + typeChar + "axpy");
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value scaleKernel = buildkernel.execute(SCALE_KERNEL, "scale", "pointer, const pointer, double, sint32");
            Value configuredScaleKernel = scaleKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            taxpy.execute(numElements, alpha, x, 1, y, 1);
            configuredScaleKernel.execute(z, x, 2, numElements);
            configuredScaleKernel.execute(y, y, 2, numElements);

            assertOutputVectorIsCorrect(numElements, z, (Integer i) -> 2 * i);
            assertOutputVectorIsCorrect(numElements, y, (Integer i) -> 2 * i);
        }
    }

    /**
     * Test a 2 independent kernels followed by a BLAS kernel followed by 2 independent kernels;
     * A--->C--->D
     * B---/ \-->E
     */
    @Test
    public void testTaxpyJoinForkPattern() {
        // x = (0, 1, 2, ..., numElements-1)
        // y = (0, 2, 4, ..., 2*(numElements-1))
        // z = (0, 0, 0, ..., 0)
        // z := 2 * x
        // y := 2 * y
        // z := -1 * z + y
        // z := -2 * z
        // y := 2 * y
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            String cudaType = "double";
            int numElements = 1000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value alpha = deviceArrayConstructor.execute(cudaType, 1);
            alpha.setArrayElement(0, -1);

            // Create some arrays;
            Value x = deviceArrayConstructor.execute(cudaType, numElements);
            Value y = deviceArrayConstructor.execute(cudaType, numElements);
            Value z = deviceArrayConstructor.execute(cudaType, numElements);
            assertEquals(numElements, x.getArraySize());
            assertEquals(numElements, y.getArraySize());
            assertEquals(numElements, z.getArraySize());

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, i);
                y.setArrayElement(i, 2 * i);
                z.setArrayElement(i, 0);
            }

            // Define kernels;
            Value taxpy = context.eval("grcuda", "BLAS::cublas" + typeChar + "axpy");
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value scaleKernel = buildkernel.execute(SCALE_KERNEL, "scale", "pointer, const pointer, double, sint32");
            Value configuredScaleKernel = scaleKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredScaleKernel.execute(z, x, 2, numElements); // z = 0, 2, 4, ...
            configuredScaleKernel.execute(y, y, 2, numElements); // y = 0, 4, 8, ...
            taxpy.execute(numElements, alpha, y, 1, z, 1); // z = 0, -2, -4, ...
            configuredScaleKernel.execute(z, z, -1, numElements); // z = 0, 2, 4, ...
            configuredScaleKernel.execute(y, y, 0.5, numElements); // y = 0, 2, 4, ...

            assertOutputVectorIsCorrect(numElements, z, (Integer i) -> 2 * i);
            assertOutputVectorIsCorrect(numElements, y, (Integer i) -> 2 * i);
        }
    }

    /**
     * Test a BLAS kernel followed by 2 independent kernels followed by a BLAS kernel;
     * A--->B--->D
     *  \-->C---/
     */
    @Test
    public void testTaxpyForkJoinPattern() {
        // x = (0, 1, 2, ..., numElements-1)
        // y = (0, 2, 4, ..., 2*(numElements-1))
        // z = (0, 0, 0, ..., 0)
        // x := -1 * x + y
        // x = (0, 1, 2, ..., numElements-1)
        // z := 2 * x
        // y := 4 * y
        // y := -1 * z + y
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            String cudaType = "double";
            int numElements = 1000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value alpha = deviceArrayConstructor.execute(cudaType, 1);
            alpha.setArrayElement(0, -1);

            // Create some arrays;
            Value x = deviceArrayConstructor.execute(cudaType, numElements);
            Value y = deviceArrayConstructor.execute(cudaType, numElements);
            Value z = deviceArrayConstructor.execute(cudaType, numElements);
            assertEquals(numElements, x.getArraySize());
            assertEquals(numElements, y.getArraySize());
            assertEquals(numElements, z.getArraySize());

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, i);
                y.setArrayElement(i, 2 * i);
                z.setArrayElement(i, 0);
            }

            // Define kernels;
            Value taxpy = context.eval("grcuda", "BLAS::cublas" + typeChar + "axpy");
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value scaleKernel = buildkernel.execute(SCALE_KERNEL, "scale", "pointer, const pointer, double, sint32");
            Value configuredScaleKernel = scaleKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            taxpy.execute(numElements, alpha, x, 1, y, 1);
            configuredScaleKernel.execute(z, x, 2, numElements);
            configuredScaleKernel.execute(y, y, 4, numElements);
            taxpy.execute(numElements, alpha, z, 1, y, 1);

            assertOutputVectorIsCorrect(numElements, z, (Integer i) -> 2 * i);
        }
    }

    /**
     * Test a BLAS kernel followed by 2 independent kernels followed by a BLAS kernel;
     *       /-->E-->F
     * A--->B--->D
     *  \-->C---/
     */
    @Test
    public void testTaxpyIndependentCompPattern() {
        // x = (0, 1, 2, ..., numElements-1)
        // y = (0, 2, 4, ..., 2*(numElements-1))
        // z = (0, 0, 0, ..., 0)
        // x := -1 * x + y
        // x = (0, 1, 2, ..., numElements-1)
        // z := 2 * x
        // y := 4 * y
        // y := -1 * z + y
        // x := 2 * x
        // x := 2 * x
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            String cudaType = "double";
            int numElements = 1000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value alpha = deviceArrayConstructor.execute(cudaType, 1);
            alpha.setArrayElement(0, -1);

            // Create some arrays;
            Value x = deviceArrayConstructor.execute(cudaType, numElements);
            Value y = deviceArrayConstructor.execute(cudaType, numElements);
            Value z = deviceArrayConstructor.execute(cudaType, numElements);
            assertEquals(numElements, x.getArraySize());
            assertEquals(numElements, y.getArraySize());
            assertEquals(numElements, z.getArraySize());

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, i);
                y.setArrayElement(i, 2 * i);
                z.setArrayElement(i, 0);
            }

            // Define kernels;
            Value taxpy = context.eval("grcuda", "BLAS::cublas" + typeChar + "axpy");
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value scaleKernel = buildkernel.execute(SCALE_KERNEL, "scale", "pointer, const pointer, double, sint32");
            Value configuredScaleKernel = scaleKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            taxpy.execute(numElements, alpha, x, 1, y, 1);
            configuredScaleKernel.execute(z, x, 2, numElements);
            configuredScaleKernel.execute(y, y, 4, numElements);
            configuredScaleKernel.execute(x, x, 2, numElements);
            configuredScaleKernel.execute(x, x, 2, numElements);
            taxpy.execute(numElements, alpha, z, 1, y, 1);

            assertOutputVectorIsCorrect(numElements, z, (Integer i) -> 2 * i);
            assertOutputVectorIsCorrect(numElements, x, (Integer i) -> 4 * i);
        }
    }

    /**
     * Validation function for vectors.
     */
    private void assertOutputVectorIsCorrect(int len, Value deviceArray,
                                             Function<Integer, Integer> outFunc) {
        for (int i = 0; i < len; i++) {
            double expected = outFunc.apply(i);
            double actual = deviceArray.getArrayElement(i).asDouble();
            assertEquals(expected, actual, 1e-5);
        }
    }
}
