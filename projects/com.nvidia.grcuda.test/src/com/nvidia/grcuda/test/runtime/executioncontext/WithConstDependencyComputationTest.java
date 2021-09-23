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

import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.mock.ArgumentMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.util.mock.KernelExecutionMock;
import com.nvidia.grcuda.test.util.mock.SyncExecutionMock;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class WithConstDependencyComputationTest {

    @Test
    public void addVertexToDAGTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(DependencyPolicyEnum.WITH_CONST);
        // Create two mock kernel executions;
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(2))).schedule();

        ExecutionDAG dag = context.getDag();

        assertEquals(1, dag.getNumVertices());
        assertEquals(0, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
        assertTrue(dag.getFrontier().get(0).isFrontier());
        assertTrue(dag.getFrontier().get(0).isStart());

        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(3))).schedule();

        assertEquals(2, dag.getNumVertices());
        assertEquals(0, dag.getNumEdges());
        assertEquals(2, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(dag.getVertices().get(0), dag.getFrontier().get(0));
        assertEquals(dag.getVertices().get(1), dag.getFrontier().get(1));
        assertTrue(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(1).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertTrue(dag.getVertices().get(1).isStart());
        // Check that no children or parents are present;
        assertEquals(0, dag.getVertices().get(0).getChildVertices().size());
        assertEquals(0, dag.getVertices().get(1).getParentVertices().size());
    }


    @Test
    public void dependencyPipelineSimpleMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(DependencyPolicyEnum.WITH_CONST);
        // Create 4 mock kernel executions. In this case, kernel 3 requires 1 and 2 to finish,
        //   and kernel 4 requires kernel 3 to finish. The final frontier is composed of kernel 3 (arguments "1" and "2" are active),
        //   and kernel 4 (argument "3" is active);
        // A(1r) -> C(1, 2) -> D(2)
        // B(1r) -/
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1, true))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1, true))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(4, dag.getNumVertices());
        assertEquals(3, dag.getNumEdges());
        assertEquals(2, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(2), dag.getVertices().get(3))),
                new HashSet<>(dag.getFrontier()));
        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertTrue(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());
        assertTrue(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        // Check if the third vertex is a child of first and second;
        assertEquals(2, dag.getVertices().get(2).getParents().size());
        assertEquals(new HashSet<>(dag.getVertices().get(2).getParentVertices()),
                new HashSet<>(Arrays.asList(dag.getVertices().get(0), dag.getVertices().get(1))));
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(0).getChildVertices().get(0));
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(1).getChildVertices().get(0));
        // Check if the fourth vertex is a child of the third;
        assertEquals(1, dag.getVertices().get(3).getParents().size());
        assertEquals(1, dag.getVertices().get(2).getChildren().size());
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(3).getParentVertices().get(0));
        assertEquals(dag.getVertices().get(3), dag.getVertices().get(2).getChildVertices().get(0));
    }

    @Test
    public void forkedComputationTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(DependencyPolicyEnum.WITH_CONST);

        // A(1) --> B(1R)
        //      \-> C(1R)
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1, true))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1, true))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(3, dag.getNumVertices());
        assertEquals(2, dag.getNumEdges());
        assertEquals(3, dag.getFrontier().size());

        assertTrue(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertTrue(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());

        assertEquals(dag.getVertices().get(0), dag.getVertices().get(1).getParentVertices().get(0));
        assertEquals(1, dag.getVertices().get(2).getParentVertices().size());
        assertFalse(dag.getVertices().get(2).getParentVertices().contains(dag.getVertices().get(1)));
        assertFalse(dag.getVertices().get(1).getChildVertices().contains(dag.getVertices().get(2)));

        // Add a fourth computation that depends on both B and C, and depends on both;
        // A(1) -> B(1R) -> D(1)
        //      \- C(1R) /
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        assertEquals(4, dag.getNumVertices());
        assertEquals(4, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
    }

    @Test
    public void complexFrontierMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(DependencyPolicyEnum.WITH_CONST);

        // A(1R,2) -> B(1) -> D(1R,3)
        //    \----> C(2R) \----> E(1R,4) -> F(4)
        // The final frontier is composed by A(2), B(1), C(2), D(1, 3), E(1), F(4);
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2, true))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(6, dag.getNumVertices());
        assertEquals(5, dag.getNumEdges());
        assertEquals(6, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(dag.getVertices()), new HashSet<>(dag.getFrontier()));

        assertTrue(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertTrue(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());
        assertTrue(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        assertTrue(dag.getVertices().get(4).isFrontier());
        assertFalse(dag.getVertices().get(4).isStart());
        assertTrue(dag.getVertices().get(5).isFrontier());
        assertFalse(dag.getVertices().get(5).isStart());
        // Check that D is a child of B and C and D are not connected;
        assertEquals(dag.getVertices().get(1), dag.getVertices().get(3).getParentVertices().get(0));
        assertFalse(dag.getVertices().get(3).getParentVertices().contains(dag.getVertices().get(2)));
        assertFalse(dag.getVertices().get(2).getChildVertices().contains(dag.getVertices().get(3)));
    }

    @Test
    public void complexFrontier2MockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST).build();

        // A(1R,2) -> B(1) -> D(1R,3) ---------> G(1,3,4)
        //         \- C(2R) \- E(1R,4) ----> F(4) -/
        // The final frontier is composed by A(2), C(2R), G(1, 3, 4);
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2, true))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(3), new ArgumentMock(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(7, dag.getNumVertices());
        assertEquals(7, dag.getNumEdges());
        assertEquals(3, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(0), dag.getVertices().get(2), dag.getVertices().get(6))),
                new HashSet<>(dag.getFrontier()));

        assertTrue(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());
        assertFalse(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        assertFalse(dag.getVertices().get(4).isFrontier());
        assertFalse(dag.getVertices().get(4).isStart());
        assertFalse(dag.getVertices().get(5).isFrontier());
        assertFalse(dag.getVertices().get(5).isStart());
        assertTrue(dag.getVertices().get(6).isFrontier());
        assertFalse(dag.getVertices().get(5).isStart());
        // Check that D is a child of B and C and D are not connected;
        assertEquals(dag.getVertices().get(1), dag.getVertices().get(3).getParentVertices().get(0));
        assertFalse(dag.getVertices().get(3).getParentVertices().contains(dag.getVertices().get(2)));
        assertFalse(dag.getVertices().get(2).getChildVertices().contains(dag.getVertices().get(3)));
        // Check that D and E are not connected;
        assertFalse(dag.getVertices().get(4).getParentVertices().contains(dag.getVertices().get(3)));
        assertFalse(dag.getVertices().get(3).getChildVertices().contains(dag.getVertices().get(4)));
        // Check that G is child exactly of D and F;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(3), dag.getVertices().get(5))), new HashSet<>(dag.getVertices().get(6).getParentVertices()));
        // Check that E and G are not connected;
        assertFalse(dag.getVertices().get(6).getParentVertices().contains(dag.getVertices().get(4)));
        assertFalse(dag.getVertices().get(4).getChildVertices().contains(dag.getVertices().get(6)));
    }

    @Test
    public void dependencyPipelineSimpleWithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST).build();
        // Create 4 mock kernel executions. In this case, kernel 3 requires 1 and 2 to finish,
        //   and kernel 4 requires kernel 3 to finish. The final frontier is composed of kernel 3 (arguments "1" and "2" are active),
        //   and kernel 4 (argument "3" is active);
        // A(1r) -> C(1, 2) -> D(2)
        // B(1r) -/
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1, true))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1, true))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(4, dag.getNumVertices());
        assertEquals(3, dag.getNumEdges());
        assertEquals(2, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(2), dag.getVertices().get(3))),
                new HashSet<>(dag.getFrontier()));
        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertTrue(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());
        assertTrue(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        // Check if the third vertex is a child of first and second;
        assertEquals(2, dag.getVertices().get(2).getParents().size());
        assertEquals(new HashSet<>(dag.getVertices().get(2).getParentVertices()),
                new HashSet<>(Arrays.asList(dag.getVertices().get(0), dag.getVertices().get(1))));
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(0).getChildVertices().get(0));
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(1).getChildVertices().get(0));
        // Check if the fourth vertex is a child of the third;
        assertEquals(1, dag.getVertices().get(3).getParents().size());
        assertEquals(1, dag.getVertices().get(2).getChildren().size());
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(3).getParentVertices().get(0));
        assertEquals(dag.getVertices().get(3), dag.getVertices().get(2).getChildVertices().get(0));

        // Finish the computation;
        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        assertEquals(0, dag.getFrontier().size());
    }

    @Test
    public void forkedComputationWithSyncTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST).build();

        // A(1) --> B(1R)
        //      \-> C(1R)
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1, true))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1, true))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(3, dag.getNumVertices());
        assertEquals(2, dag.getNumEdges());
        assertEquals(3, dag.getFrontier().size());

        assertTrue(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertTrue(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());

        assertEquals(dag.getVertices().get(0), dag.getVertices().get(1).getParentVertices().get(0));
        assertEquals(1, dag.getVertices().get(2).getParentVertices().size());
        assertFalse(dag.getVertices().get(2).getParentVertices().contains(dag.getVertices().get(1)));
        assertFalse(dag.getVertices().get(1).getChildVertices().contains(dag.getVertices().get(2)));

        // Add a fourth computation that depends on both B and C, and depends on both;
        // A(1) -> B(1R) -> D(1)
        //     \-> C(1R) /
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        assertEquals(4, dag.getNumVertices());
        assertEquals(4, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());

        // Finish the computation;
        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        assertEquals(0, dag.getFrontier().size());
    }

    @Test
    public void complexFrontierWithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(DependencyPolicyEnum.WITH_CONST,
                RetrieveNewStreamPolicyEnum.FIFO, RetrieveParentStreamPolicyEnum.DISJOINT);

        // A(1R,2) -> B(1) ---> D(1R,3)
        //        \-> C(2R) \-> E(1R,4) -> F(4)
        // The final frontier is composed by  A(1R,2), B(1), C(2R), D(1R,3), E(1R,4), F(4);
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2, true))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(6, dag.getNumVertices());
        assertEquals(5, dag.getNumEdges());
        assertEquals(6, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(dag.getVertices()),
                new HashSet<>(dag.getFrontier()));

        assertTrue(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertTrue(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());
        assertTrue(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        assertTrue(dag.getVertices().get(4).isFrontier());
        assertFalse(dag.getVertices().get(4).isStart());
        assertTrue(dag.getVertices().get(5).isFrontier());
        assertFalse(dag.getVertices().get(5).isStart());
        // Check that D is a child of B and C and D are not connected;
        assertEquals(dag.getVertices().get(1), dag.getVertices().get(3).getParentVertices().get(0));
        assertFalse(dag.getVertices().get(3).getParentVertices().contains(dag.getVertices().get(2)));
        assertFalse(dag.getVertices().get(2).getChildVertices().contains(dag.getVertices().get(3)));
        // Check that D and E are not connected;
        assertFalse(dag.getVertices().get(4).getParentVertices().contains(dag.getVertices().get(3)));
        assertFalse(dag.getVertices().get(3).getChildVertices().contains(dag.getVertices().get(4)));

        // Synchronize computations;
        // A(1R,2) -> B(1) ---> D(1R,3)
        //        \-> C(2R) \-> E(1R,4) -> F(4)
        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        assertEquals(4, dag.getFrontier().size());

        // Note that syncing F(4) will also sync B(1) although it's on a different stream;
        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();
        assertEquals(1, dag.getFrontier().size());

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(3))).schedule();
        assertEquals(0, dag.getFrontier().size());
    }

    @Test
    public void complexFrontier2WithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(DependencyPolicyEnum.WITH_CONST,
                RetrieveNewStreamPolicyEnum.FIFO, RetrieveParentStreamPolicyEnum.DISJOINT);

        // A(1R,2) -> B(1) -> D(1R,3) ---------> G(1, 3, 4)
        //        \-> C(2R) \-> E(1R,4) -> F(4) -/
        // The final frontier is composed by A(1R,2), C(2R), G(1, 3, 4);
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2, true))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(3), new ArgumentMock(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(7, dag.getNumVertices());
        assertEquals(7, dag.getNumEdges());
        assertEquals(3, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(0), dag.getVertices().get(2), dag.getVertices().get(6))),
                new HashSet<>(dag.getFrontier()));

        assertTrue(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());
        assertFalse(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        assertFalse(dag.getVertices().get(4).isFrontier());
        assertFalse(dag.getVertices().get(4).isStart());
        assertFalse(dag.getVertices().get(5).isFrontier());
        assertFalse(dag.getVertices().get(5).isStart());
        assertTrue(dag.getVertices().get(6).isFrontier());
        assertFalse(dag.getVertices().get(6).isStart());
        // Check that D is a child of B and C and D are not connected;
        assertEquals(dag.getVertices().get(1), dag.getVertices().get(3).getParentVertices().get(0));
        assertFalse(dag.getVertices().get(3).getParentVertices().contains(dag.getVertices().get(2)));
        assertFalse(dag.getVertices().get(2).getChildVertices().contains(dag.getVertices().get(3)));
        // Check that D and E are not connected;
        assertFalse(dag.getVertices().get(4).getParentVertices().contains(dag.getVertices().get(3)));
        assertFalse(dag.getVertices().get(3).getChildVertices().contains(dag.getVertices().get(4)));
        // Check that G is child exactly of D and F;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(3), dag.getVertices().get(5))), new HashSet<>(dag.getVertices().get(6).getParentVertices()));

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        assertEquals(1, dag.getFrontier().size());

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();
        assertEquals(0, dag.getFrontier().size());
    }
}
