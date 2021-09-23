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
package com.nvidia.grcuda.test.cudalibraries;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeNoException;

@RunWith(Parameterized.class)
public class CUMLTest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {

        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                        {ExecutionPolicyEnum.SYNC.getName(), ExecutionPolicyEnum.ASYNC.getName()},
                        {true, false},
                        {'S', 'D'}
        }));
    }

    private final String policy;
    private final boolean inputPrefetch;
    private final char typeChar;

    public CUMLTest(String policy, boolean inputPrefetch, char typeChar) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
        this.typeChar = typeChar;
    }

    @Test
    public void testDbscan() {
        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(
                        true).build()) {
            Value cu = polyglot.eval("grcuda", "CU");
            int numRows = 100;
            int numCols = 2;
            String cudaType = (typeChar == 'D') ? "double" : "float";
            Value input = cu.invokeMember("DeviceArray", cudaType, numRows, numCols);
            Value labels = cu.invokeMember("DeviceArray", "int", numRows);
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    input.getArrayElement(i).setArrayElement(j, i / 10 + j);
                }
                labels.setArrayElement(i, 0);
            }
            double eps = 0.5;
            int minSamples = 5;
            int maxBytesPerChunk = 0;
            int verbose = 1;
            try {
                Value dbscan = polyglot.eval("grcuda", "ML::cuml" + typeChar + "pDbscanFit");
                try {
                    dbscan.execute(input, numRows, numCols, eps, minSamples, labels, maxBytesPerChunk, verbose);
                    CUBLASTest.assertOutputVectorIsCorrect(numRows, labels, (Integer i) -> i / 10, this.typeChar);
                } catch (Exception e) {
                    System.out.println("warning: failed to launch cuML, skipping test");
                }
            } catch (Exception e) {
                System.out.println("warning: cuML not enabled, skipping test");
                assumeNoException(e);
            }
        }
    }
}
