/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
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
package com.nvidia.grcuda.test.cudalibraries;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;
import java.util.function.Function;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

@RunWith(Parameterized.class)
public class CUBLASTest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {

        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {ExecutionPolicyEnum.SYNC.getName(), ExecutionPolicyEnum.ASYNC.getName()},
                {true, false},
                {'S', 'D', 'C', 'Z'}
        }));
    }

    private final String policy;
    private final boolean inputPrefetch;
    private final char typeChar;

    public CUBLASTest(String policy, boolean inputPrefetch, char typeChar) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
        this.typeChar = typeChar;
    }

    /**
     * BLAS Level-1 Test.
     */
    @Test
    public void testTaxpy() {
        // x = (0, 1, 2, ..., numDim-1)
        // y = (0, -2, -4, ..., -2*numDim-2)
        // y := -1 * x + y
        // y = (0, 1, 2, ..., numDim-1)
        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy)
                .option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(true).build()) {
            Value cu = polyglot.eval("grcuda", "CU");
            boolean isComplex = (typeChar == 'C') || (typeChar == 'Z');
            String cudaType = ((typeChar == 'D') || (typeChar == 'Z')) ? "double" : "float";
            int numDim = 1000;
            int numElements = isComplex ? numDim * 2 : numDim;
            Value alpha = cu.invokeMember("DeviceArray", cudaType, isComplex ? 2 : 1);
            alpha.setArrayElement(0, -1);
            if (isComplex) {
                alpha.setArrayElement(1, 0);
            }
            Value x = cu.invokeMember("DeviceArray", cudaType, numElements);
            Value y = cu.invokeMember("DeviceArray", cudaType, numElements);
            assertEquals(numElements, x.getArraySize());
            assertEquals(numElements, y.getArraySize());

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, i);
                y.setArrayElement(i, 2 * i);
            }
            Value taxpy = polyglot.eval("grcuda", "BLAS::cublas" + typeChar + "axpy");
            taxpy.execute(numDim, alpha, x, 1, y, 1);
            assertOutputVectorIsCorrect(numElements, y, (Integer i) -> i);
        }
    }

    /**
     * BLAS Level-2 Test.
     */
    @Test
    public void testTgemv() {
        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy)
                .option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(true).build()) {
            Value cu = polyglot.eval("grcuda", "CU");
            int numDim = 10;
            boolean isComplex = (typeChar == 'C') || (typeChar == 'Z');
            String cudaType = ((typeChar == 'D') || (typeChar == 'Z')) ? "double" : "float";
            int numElements = isComplex ? numDim * 2 : numDim;
            Value alpha = cu.invokeMember("DeviceArray", cudaType, isComplex ? 2 : 1);
            Value beta = cu.invokeMember("DeviceArray", cudaType, isComplex ? 2 : 1);
            alpha.setArrayElement(0, -1);
            beta.setArrayElement(0, 2);
            if (isComplex) {
                alpha.setArrayElement(1, 0);
                beta.setArrayElement(1, 0);
            }

            // complex types require two elements along 1st dimension (since column-major order)
            Value matrixA = cu.invokeMember("DeviceArray", cudaType, numElements, numDim, "F");
            Value x = cu.invokeMember("DeviceArray", cudaType, numElements);
            Value y = cu.invokeMember("DeviceArray", cudaType, numElements);

            // set matrix
            // A: identity matrix
            for (int j = 0; j < numDim; j++) {
                for (int i = 0; i < numElements; i++) {
                    // complex types require two elements along 1st dimension (since column-major order)
                    Value row = matrixA.getArrayElement(i);
                    row.setArrayElement(j, ((!isComplex & (i == j)) || (isComplex && (i == (2 * j)))) ? 1.0 : 0.0);
                }
            }

            // set vectors
            // x = (1, 2, ..., numDim)
            // y = (1, 2, ..., numDim)
            for (int i = 0; i < numElements; i++) {
                x.setArrayElement(i, i);
                y.setArrayElement(i, i);
            }
            Value tgemv = polyglot.eval("grcuda", "BLAS::cublas" + typeChar + "gemv");
            final int cublasOpN = 0;
            tgemv.execute(cublasOpN, numDim, numDim,
                    alpha,
                    matrixA, numDim,
                    x, 1,
                    beta,
                    y, 1);
            assertOutputVectorIsCorrect(numElements, y, (Integer i) -> i);
        }
    }

    /**
     * BLAS Level-3 Test.
     */
    @Test
    public void testTgemm() {
        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy)
                .option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(true).build()) {
            Value cu = polyglot.eval("grcuda", "CU");
            int numDim = 10;
            boolean isComplex = (typeChar == 'C') || (typeChar == 'Z');
            String cudaType = ((typeChar == 'D') || (typeChar == 'Z')) ? "double" : "float";
            int numElements = isComplex ? numDim * 2 : numDim;
            Value alpha = cu.invokeMember("DeviceArray", cudaType, isComplex ? 2 : 1);
            Value beta = cu.invokeMember("DeviceArray", cudaType, isComplex ? 2 : 1);
            alpha.setArrayElement(0, -1);
            beta.setArrayElement(0, 2);
            if (isComplex) {
                alpha.setArrayElement(1, 0);
                beta.setArrayElement(1, 0);
            }

            // complex types require two elements along 1st dimension (since column-major order)
            Value matrixA = cu.invokeMember("DeviceArray", cudaType, numElements, numDim, "F");
            Value matrixB = cu.invokeMember("DeviceArray", cudaType, numElements, numDim, "F");
            Value matrixC = cu.invokeMember("DeviceArray", cudaType, numElements, numDim, "F");

            // set matrix
            // A: identity matrix
            for (int j = 0; j < numDim; j++) {
                for (int i = 0; i < numElements; i++) {
                    // complex types require two elements along 1st dimension (since column-major order)
                    Value row = matrixA.getArrayElement(i);
                    row.setArrayElement(j, ((!isComplex & (i == j)) || (isComplex && (i == (2 * j)))) ? 1.0 : 0.0);
                }
            }
            // B == C
            for (int j = 0; j < numDim; j++) {
                for (int i = 0; i < numElements; i++) {
                    Value row = matrixB.getArrayElement(i);
                    row.setArrayElement(j, i + numElements * j);
                }
            }
            for (int j = 0; j < numDim; j++) {
                for (int i = 0; i < numElements; i++) {
                    Value row = matrixC.getArrayElement(i);
                    row.setArrayElement(j, i + numElements * j);
                }
            }
            Value tgemm = polyglot.eval("grcuda", "BLAS::cublas" + typeChar + "gemm");
            final int cublasOpN = 0;
            tgemm.execute(cublasOpN, cublasOpN, numDim, numDim, numDim,
                    alpha,
                    matrixA, numDim,
                    matrixB, numDim,
                    beta,
                    matrixC, numDim);
            assertOutputMatrixIsCorrect(numDim, numElements, matrixC, (Integer i) -> i);
        }
    }

    /**
     * Validation function for vectors.
     */
    public static void assertOutputVectorIsCorrect(int len, Value deviceArray,
                    Function<Integer, Integer> outFunc, char typeChar) {
        boolean hasDouble = (typeChar == 'D') || (typeChar == 'Z');
        for (int i = 0; i < len; i++) {
            if (hasDouble) {
                double expected = outFunc.apply(i);
                double actual = deviceArray.getArrayElement(i).asDouble();
                assertEquals(expected, actual, 1e-5);
            } else {
                float expected = outFunc.apply(i);
                float actual = deviceArray.getArrayElement(i).asFloat();
                assertEquals(expected, actual, 1e-5f);
            }
        }
    }

    private void assertOutputVectorIsCorrect(int len, Value deviceArray,
                                                    Function<Integer, Integer> outFunc) {
        CUBLASTest.assertOutputVectorIsCorrect(len, deviceArray, outFunc, this.typeChar);
    }

    /**
     * Validation function for matrix.
     */
    private void assertOutputMatrixIsCorrect(int numDim, int numElements, Value matrix,
                    Function<Integer, Integer> outFunc) {
        boolean hasDouble = (typeChar == 'D') || (typeChar == 'Z');
        for (int j = 0; j < numDim; j++) {
            for (int i = 0; i < numElements; i++) {
                int idx = i + numElements * j;
                if (hasDouble) {
                    double expected = outFunc.apply(idx);
                    double actual = matrix.getArrayElement(i).getArrayElement(j).asDouble();
                    assertEquals(expected, actual, 1e-5);
                } else {
                    float expected = outFunc.apply(idx);
                    float actual = matrix.getArrayElement(i).getArrayElement(j).asFloat();
                    assertEquals(expected, actual, 1e-5f);
                }
            }
        }
    }
}
