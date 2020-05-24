package com.nvidia.grcuda.test.gpu.executioncontext;

import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.test.mock.ArgumentMock;
import com.nvidia.grcuda.test.mock.KernelExecutionMock;
import com.nvidia.grcuda.test.mock.MultithreadGrCUDAExecutionContextMock;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class MultithreadGrCUDAExecutionContextTest {

    @Test
    public void multithreadAddVertexToDAGTest() throws UnsupportedTypeException {
        MultithreadGrCUDAExecutionContextMock context = new MultithreadGrCUDAExecutionContextMock();
        // Create two mock kernel executions;
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2)), 100).schedule();

        ExecutionDAG dag = context.getDag();

        assertEquals(1, dag.getNumVertices());
        assertEquals(0, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
        assertTrue(dag.getFrontier().get(0).isFrontier());
        assertTrue(dag.getFrontier().get(0).isStart());

        KernelExecutionMock k2 = new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2)), 100);
        k2.schedule();

        assertEquals(2, dag.getNumVertices());
        assertEquals(1, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(dag.getVertices().get(1), dag.getFrontier().get(0));
        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(1).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isStart());
        // Check if the first vertex is a parent of the second;
        assertEquals(dag.getVertices().get(0), dag.getVertices().get(1).getParentVertices().get(0));
        // Check if the second vertex is a child of the first;
        assertEquals(dag.getVertices().get(1), dag.getVertices().get(0).getChildVertices().get(0));

        context.waitFinish(k2);
        assertEquals(0, dag.getFrontier().size());
    }

    @Test
    public void dependencyPipelineSimpleMockTest() throws UnsupportedTypeException {
        MultithreadGrCUDAExecutionContextMock context = new MultithreadGrCUDAExecutionContextMock();
        // Create 4 mock kernel executions. In this case, kernel 3 requires 1 and 2 to finish,
        //   and kernel 4 requires kernel 3 to finish. The final frontier is composed of kernel 3 (arguments "1" and "2" are active),
        //   and kernel 4 (argument "3" is active);
        // A(1) -> C(1,2,3) -> D(3)
        // B(2) /
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1)), 100).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2)), 100).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2), new ArgumentMock(3)), 100).schedule();
        KernelExecutionMock k3 = new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(3)), 100);
        k3.schedule();

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

        context.waitFinish(k3);
        assertEquals(0, dag.getFrontier().size());
    }

    @Test
    public void complexFrontierMockTest() throws UnsupportedTypeException, InterruptedException {
        MultithreadGrCUDAExecutionContextMock context = new MultithreadGrCUDAExecutionContextMock();

        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //   \ C(2)
        KernelExecutionMock k1 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(2)), 100);
        k1.schedule();
        KernelExecutionMock k2 = new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1)), 200);
        k2.schedule();
        KernelExecutionMock k3 = new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2)), 100);
        k3.schedule();
        KernelExecutionMock k4 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(3)), 100);
        k4.schedule();
        KernelExecutionMock k5 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(4)), 100);
        k5.schedule();
        KernelExecutionMock k6 = new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4)), 100);
        k6.schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(6, dag.getNumVertices());
        assertEquals(5, dag.getNumEdges());
        assertEquals(4, dag.getFrontier().size());

        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(2), dag.getVertices().get(3), dag.getVertices().get(4), dag.getVertices().get(5))),
                new HashSet<>(dag.getFrontier()));

        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());
        assertTrue(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        assertTrue(dag.getVertices().get(4).isFrontier());
        assertFalse(dag.getVertices().get(4).isStart());
        assertTrue(dag.getVertices().get(5).isFrontier());
        assertFalse(dag.getVertices().get(5).isStart());

        context.waitFinish(k3);
        context.waitFinish(k6);

        // Check the DAG structure;
        assertEquals(6, dag.getNumVertices());
        assertEquals(5, dag.getNumEdges());
        assertEquals(0, dag.getFrontier().size());
    }

}
