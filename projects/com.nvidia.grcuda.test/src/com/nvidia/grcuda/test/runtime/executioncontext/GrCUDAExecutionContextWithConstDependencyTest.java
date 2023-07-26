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
package com.nvidia.grcuda.test.runtime.executioncontext;

import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.test.util.GrCUDATestOptionsStruct;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.graalvm.polyglot.Context;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

@RunWith(Parameterized.class)
public class GrCUDAExecutionContextWithConstDependencyTest {

    /**
     * Tests are executed for each of the {@link AsyncGrCUDAExecutionContext} values;
     * @return the current stream policy
     */

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return GrCUDATestUtil.getAllOptionCombinationsSingleGPU();
    }

    private final GrCUDATestOptionsStruct options;

    public GrCUDAExecutionContextWithConstDependencyTest(GrCUDATestOptionsStruct options) {
        this.options = options;
    }

    @Test
    public void parallelKernelsWithReadOnlyArgsTest() {
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            GrCUDAComputationsWithGPU.parallelKernelsWithReadOnlyArgs(context);
        }
    }

    @Test
    public void simpleForkReadInputTest() {
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            GrCUDAComputationsWithGPU.simpleForkReadInput(context);
        }
    }

    @Test
    public void forkWithReadOnlyTest() {
        // Test a computation of form A(1) --> B(1r, 2)
        //                                 \-> C(1r, 3)
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            GrCUDAComputationsWithGPU.forkWithReadOnly(context);
        }
    }

    @Test
    public void dependencyPipelineDiamondTest() {
        // Test a computation of form A(1) --> B(1r, 2) -> D(1)
        //                                 \-> C(1r, 3) -/
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            GrCUDAComputationsWithGPU.dependencyPipelineDiamond(context);
        }
    }

    @Test
    public void joinWithExtraKernelTest() {
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            GrCUDAComputationsWithGPU.joinWithExtraKernel(context);
        }
    }
}
