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

import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.executioncontext.GraphExport;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.mock.*;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

import com.nvidia.grcuda.test.util.mock.ArgumentMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.util.mock.KernelExecutionMock;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;


@RunWith(Parameterized.class)
public class ExecutionDAGExportTest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
                {RetrieveNewStreamPolicyEnum.ALWAYS_NEW},
                {RetrieveNewStreamPolicyEnum.REUSE},
        });
    }

    private final RetrieveNewStreamPolicyEnum policy;

    public ExecutionDAGExportTest(RetrieveNewStreamPolicyEnum policy) {
        this.policy = policy;
    }

    @Test
    public void complexFrontierExportTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new AsyncGrCUDAExecutionContextMock();

        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //    \----> C(2)
        // The final frontier is composed by C(2), D(3), E(1), F(4);
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();

        ExecutionDAG dag = context.getDag();

        GraphExport graphExport = new GraphExport(dag);
        //graphExport.graphGenerator("../graphComplexFrontierExportTest");
    }

    @Test
    public void streamSelection2ExportTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveNewStreamPolicy(this.policy).build();

        // A(1,2) -> B(1) -> D(1,3)
        //    \----> C(2)
        // E(4) -> F(4, 5)
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(4), new ArgumentMock(5))).schedule();

        ExecutionDAG dag = context.getDag();
        GraphExport graphExport = new GraphExport(dag);

//        if (policy==RetrieveNewStreamPolicyEnum.ALWAYS_NEW){
//            graphExport.graphGenerator("../streamSelection2ExportTestAlwaysNew");
//        } else {
//            graphExport.graphGenerator("../streamSelection2ExportTestReuse");
//        }

    }

    @Test
    public void streamSelectionSimpleWithSyncExportTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveNewStreamPolicy(this.policy).build();
        // Create 4 mock kernel executions. In this case, kernel 3 requires 1 and 2 to finish,
        //   and kernel 4 requires kernel 3 to finish. The final frontier is composed of kernel 3 (arguments "1" and "2" are active),
        //   and kernel 4 (argument "3" is active);
        // A(1) -> C(1, 2, 3) -> D(3)
        // B(2) /
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1),
                        new ArgumentMock(2),
                        new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(3))).schedule();

        ExecutionDAG dag = context.getDag();
        GraphExport graphExport = new GraphExport(dag);


//        if (policy==RetrieveNewStreamPolicyEnum.ALWAYS_NEW){
//            graphExport.graphGenerator("../streamSelectionSimpleWithSyncExportTestAlwaysNew");
//        } else {
//            graphExport.graphGenerator("../streamSelectionSimpleWithSyncExportTestReuse");
//        }

    }

    @Test
    public void disjointArgumentStreamCross2Test() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.policy).setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT).build();

        // A(1,2,7) -> D(1,3,5)
        //          X
        // B(3,4,8) -> E(2,4,6)
        //          X
        // C(5,6,9) -> F(7,8,9)
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2), new ArgumentMock(7))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(3), new ArgumentMock(4), new ArgumentMock(8))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(5), new ArgumentMock(6), new ArgumentMock(9))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(3), new ArgumentMock(5))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(2), new ArgumentMock(4), new ArgumentMock(6))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(7), new ArgumentMock(8), new ArgumentMock(9))).schedule();

        ExecutionDAG dag = context.getDag();
        GraphExport graphExport = new GraphExport(dag);

//        if (policy==RetrieveNewStreamPolicyEnum.ALWAYS_NEW){
//            graphExport.graphGenerator("../disjointArgumentStreamCross2TestAlwaysNew");
//        } else {
//            graphExport.graphGenerator("../disjointArgumentStreamCross2TestReuse");
//        }
    }

    @Test
    public void syncParentsOfParentsTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.policy).setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT).build();

        // A(1,2) -> B(1)
        //       \-> C(2,3) -> D(2)
        //                 \-> E(3)
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(2), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(3))).schedule();

        ExecutionDAG dag = context.getDag();
        GraphExport graphExport = new GraphExport(dag);

//        if (policy==RetrieveNewStreamPolicyEnum.ALWAYS_NEW){
//            graphExport.graphGenerator("../syncParentsOfParentsTestAlwaysNew");
//        } else {
//            graphExport.graphGenerator("../syncParentsOfParentsTestReuse");
//        }
    }


}
