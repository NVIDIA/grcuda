package com.nvidia.grcuda.test.gpu.stream;

import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.GrCUDAExecutionContext;
import com.nvidia.grcuda.test.gpu.ExecutionDAGTest;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class GrCUDAStreamManagerTest {
    @Test
    public void streamSelectionSimpleMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new ExecutionDAGTest.GrCUDAExecutionContextTest();
        // Create 4 mock kernel executions. In this case, kernel 3 requires 1 and 2 to finish,
        //   and kernel 4 requires kernel 3 to finish. The final frontier is composed of kernel 3 (arguments "1" and "2" are active),
        //   and kernel 4 (argument "3" is active);
        new ExecutionDAGTest.KernelExecutionTest(context, Collections.singletonList(1)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Collections.singletonList(2)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Arrays.asList(1, 2, 3)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Collections.singletonList(3)).schedule();

        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(2, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
    }

    @Test
    public void streamSelectionMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new ExecutionDAGTest.GrCUDAExecutionContextTest();

        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //    \----> C(2)
        // The final frontier is composed by C(2), D(3), E(1), F(4);
        new ExecutionDAGTest.KernelExecutionTest(context, Arrays.asList(1, 2)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Collections.singletonList(1)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Collections.singletonList(2)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Arrays.asList(1, 3)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Arrays.asList(1, 4)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Collections.singletonList(4)).schedule();

        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(1, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(4).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(5).getComputation().getStream().getStreamNumber());
    }

    @Test
    public void streamSelection2MockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new ExecutionDAGTest.GrCUDAExecutionContextTest();

        // A(1,2) -> B(1) -> D(1,3)
        //    \----> C(2)
        // E(4) -> F(4, 5)
        new ExecutionDAGTest.KernelExecutionTest(context, Arrays.asList(1, 2)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Collections.singletonList(1)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Collections.singletonList(2)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Arrays.asList(1, 3)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Collections.singletonList(4)).schedule();
        new ExecutionDAGTest.KernelExecutionTest(context, Arrays.asList(4, 5)).schedule();


        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(2, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(4).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(5).getComputation().getStream().getStreamNumber());
    }
}
