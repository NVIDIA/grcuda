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
package com.nvidia.grcuda.test.runtime.array;

import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class MultiDimArrayTest {

    @Test
    public void test2DimArrayRowMajorFromConstructor() {
        // 2-dimensional array through DeviceArray constructor (row-major)
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceArrayContructor = context.eval("grcuda", "DeviceArray");
            final int numDim1 = 19;
            final int numDim2 = 53;
            Value matrix = deviceArrayContructor.execute("int", numDim1, numDim2);
            assertEquals(numDim1, matrix.getArraySize());
            assertEquals(numDim2, matrix.getArrayElement(0).getArraySize());
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    matrix.getArrayElement(i).setArrayElement(j, i * numDim2 + j);
                }
            }
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    assertEquals(i * numDim2 + j, matrix.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void test2DimArrayRowMajorFromPolyglotExpr() {
        // 2-dimensional array through polyglot expression "int[19][53]"
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            final int numDim1 = 19;
            final int numDim2 = 53;
            String code = String.format("int[%d][%d]", numDim1, numDim2);
            Value matrix = context.eval("grcuda", code);
            assertEquals(numDim1, matrix.getArraySize());
            assertEquals(numDim2, matrix.getArrayElement(0).getArraySize());
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    matrix.getArrayElement(i).setArrayElement(j, i * numDim2 + j);
                }
            }
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    assertEquals(i * numDim2 + j, matrix.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void test2DimArrayColMajorFromConstructor() {
        // 2-dimensional array through DeviceArray constructor (column-major)
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            final int numDim1 = 19;
            final int numDim2 = 53;
            Value matrix = deviceArrayConstructor.execute("int", numDim1, numDim2, "F");
            assertEquals(numDim1, matrix.getArraySize());
            assertEquals(numDim2, matrix.getArrayElement(0).getArraySize());
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    matrix.getArrayElement(i).setArrayElement(j, i + j * numDim1);
                }
            }
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    assertEquals(i + j * numDim1, matrix.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void test3DimArrayRowMajorFromConstructor() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            // 3-dimensional array through DeviceArray constructor (row-major)
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            final int numDim1 = 5;
            final int numDim2 = 3;
            final int numDim3 = 2;
            Value matrix = deviceArrayConstructor.execute("int", numDim1, numDim2, numDim3);
            assertEquals(numDim1, matrix.getArraySize());
            assertEquals(numDim2, matrix.getArrayElement(0).getArraySize());
            assertEquals(numDim3, matrix.getArrayElement(0).getArrayElement(0).getArraySize());
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    for (int k = 0; k < numDim3; k++) {
                        matrix.getArrayElement(i).getArrayElement(j).setArrayElement(k, i * numDim3 * numDim2 + j * numDim3 + k);
                    }
                }
            }
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    for (int k = 0; k < numDim3; k++) {
                        assertEquals(i * numDim3 * numDim2 + j * numDim3 + k,
                                        matrix.getArrayElement(i).getArrayElement(j).getArrayElement(k).asInt());
                    }
                }
            }
        }
    }

    @Test
    public void test3DimArrayColMajorFromConstructor() {
        // 3-dimensional array through DeviceArray constructor (column-major)
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            final int numDim1 = 5;
            final int numDim2 = 3;
            final int numDim3 = 2;
            Value matrix = deviceArrayConstructor.execute("int", numDim1, numDim2, numDim3, "F");
            assertEquals(numDim1, matrix.getArraySize());
            assertEquals(numDim2, matrix.getArrayElement(0).getArraySize());
            assertEquals(numDim3, matrix.getArrayElement(0).getArrayElement(0).getArraySize());
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    for (int k = 0; k < numDim3; k++) {
                        matrix.getArrayElement(i).getArrayElement(j).setArrayElement(k, i + j * numDim1 + k * numDim1 * numDim2);
                    }
                }
            }
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    for (int k = 0; k < numDim3; k++) {
                        assertEquals(i + j * numDim1 + k * numDim1 * numDim2,
                                        matrix.getArrayElement(i).getArrayElement(j).getArrayElement(k).asInt());
                    }
                }
            }
        }
    }

    @Test
    public void test4DimArrayRowMajorFromConstructor() {
        // 4-dimensional array through DeviceArray constructor (row-major)
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            final int numDim1 = 7;
            final int numDim2 = 5;
            final int numDim3 = 3;
            final int numDim4 = 2;
            Value matrix = deviceArrayConstructor.execute("int", numDim1, numDim2, numDim3, numDim4);
            assertEquals(numDim1, matrix.getArraySize());
            assertEquals(numDim2, matrix.getArrayElement(0).getArraySize());
            assertEquals(numDim3, matrix.getArrayElement(0).getArrayElement(0).getArraySize());
            assertEquals(numDim4, matrix.getArrayElement(0).getArrayElement(0).getArrayElement(0).getArraySize());
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    for (int k = 0; k < numDim3; k++) {
                        for (int l = 0; l < numDim4; l++) {
                            matrix.getArrayElement(i).getArrayElement(j).getArrayElement(k).setArrayElement(l,
                                            i * numDim4 * numDim3 * numDim2 + j * numDim4 * numDim3 + k * numDim4 + l);
                        }
                    }
                }
            }
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    for (int k = 0; k < numDim3; k++) {
                        for (int l = 0; l < numDim4; l++) {
                            assertEquals(i * numDim4 * numDim3 * numDim2 + j * numDim4 * numDim3 + k * numDim4 + l,
                                            matrix.getArrayElement(i).getArrayElement(j).getArrayElement(k).getArrayElement(l).asInt());
                        }
                    }
                }
            }
        }
    }

    @Test
    public void test4DimArrayColMajorFromConstructor() {
        // 4-dimensional array through DeviceArray constructor (column-major)
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            final int numDim1 = 7;
            final int numDim2 = 5;
            final int numDim3 = 3;
            final int numDim4 = 2;
            Value matrix = deviceArrayConstructor.execute("int", numDim1, numDim2, numDim3, numDim4, "F");
            assertEquals(numDim1, matrix.getArraySize());
            assertEquals(numDim2, matrix.getArrayElement(0).getArraySize());
            assertEquals(numDim3, matrix.getArrayElement(0).getArrayElement(0).getArraySize());
            assertEquals(numDim4, matrix.getArrayElement(0).getArrayElement(0).getArrayElement(0).getArraySize());
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    for (int k = 0; k < numDim3; k++) {
                        for (int l = 0; l < numDim4; l++) {
                            matrix.getArrayElement(i).getArrayElement(j).getArrayElement(k).setArrayElement(l,
                                            i + j * numDim1 + k * numDim1 * numDim2 + l * numDim1 * numDim2 * numDim3);
                        }
                    }
                }
            }
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    for (int k = 0; k < numDim3; k++) {
                        for (int l = 0; l < numDim4; l++) {
                            assertEquals(i + j * numDim1 + k * numDim1 * numDim2 + l * numDim1 * numDim2 * numDim3,
                                            matrix.getArrayElement(i).getArrayElement(j).getArrayElement(k).getArrayElement(l).asInt());
                        }
                    }
                }
            }
        }
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void test2DimArrayOutOfBoundsOnReadAccess() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            final int numDim1 = 19;
            final int numDim2 = 53;
            Value matrix = deviceArrayConstructor.execute("int", numDim1, numDim2);
            assertEquals(numDim1, matrix.getArraySize());
            assertEquals(numDim2, matrix.getArrayElement(0).getArraySize());
            // out-of-bounds read access
            matrix.getArrayElement(numDim1);
        }
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void test2DimArrayOutOfBoundsOnWriteAccess() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            final int numDim1 = 19;
            final int numDim2 = 53;
            Value matrix = deviceArrayConstructor.execute("int", numDim1, numDim2);
            assertEquals(numDim1, matrix.getArraySize());
            assertEquals(numDim2, matrix.getArrayElement(0).getArraySize());
            // out-of-bounds write access
            matrix.getArrayElement(0).setArrayElement(53, 42);
        }
    }

    /** CUDA C++ source code of incrementing kernel. */
    private static final String INC2D_KERNEL_SOURCE = "template <typename T>                         \n" +
                    "__global__ void inc2d(T *matrix, int num_dim1, int num_dim2) {                  \n" +
                    "  const int num_elements = num_dim1 * num_dim2;                                 \n" +
                    "  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;    \n" +
                    "       idx += gridDim.x * blockDim.x) {                                         \n" +
                    "    matrix[idx] += (T{} + 1);                                                   \n" +
                    "  }                                                                             \n" +
                    "}\n";
    /** NFI Signature of incrementing kernel. */
    private static final String INC2D_KERNEL_SIGNATURE = "pointer, sint32, sint32";

    @Test
    public void test2DimArrayAsKernelArgument() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            final Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            final int numDim1 = 19;
            final int numDim2 = 53;
            Value matrix = deviceArrayConstructor.execute("int", numDim1, numDim2);
            assertEquals(numDim1, matrix.getArraySize());
            assertEquals(numDim2, matrix.getArrayElement(0).getArraySize());
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    matrix.getArrayElement(i).setArrayElement(j, i * numDim2 + j);
                }
            }

            final Value buildKernel = context.eval("grcuda", "buildkernel");
            final Value kernel = buildKernel.execute(INC2D_KERNEL_SOURCE, "inc2d<int>", INC2D_KERNEL_SIGNATURE);
            final int blocks = 80;
            final int threadsPerBlock = 256;
            kernel.execute(blocks, threadsPerBlock).execute(matrix, numDim1, numDim2);
            for (int i = 0; i < numDim1; i++) {
                for (int j = 0; j < numDim2; j++) {
                    assertEquals(i * numDim2 + j + 1, matrix.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }
}
