package com.nvidia.grcuda.test.gpu.stream;

import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.test.mock.GrCUDAExecutionContextTest;
import com.nvidia.grcuda.test.mock.KernelExecutionTest;
import com.nvidia.grcuda.test.mock.MockArgument;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class GrCUDAStreamManagerTest {
    @Test
    public void streamSelectionSimpleMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextTest();
        // Create 4 mock kernel executions. In this case, kernel 3 requires 1 and 2 to finish,
        //   and kernel 4 requires kernel 3 to finish. The final frontier is composed of kernel 3 (arguments "1" and "2" are active),
        //   and kernel 4 (argument "3" is active);
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(1))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(2))).schedule();
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1),
                        new MockArgument(2),
                        new MockArgument(3))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(3))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(2, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(3, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(0).getComputation().getStream()));
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(1).getComputation().getStream()));
    }

    @Test
    public void streamSelectionMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextTest();

        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //    \----> C(2)
        // The final frontier is composed by C(2), D(3), E(1), F(4);
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1), new MockArgument(2))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(1))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(2))).schedule();
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1), new MockArgument(3))).schedule();
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1), new MockArgument(4))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(1, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(4).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(5).getComputation().getStream().getStreamNumber());
        assertEquals(6, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(0).getComputation().getStream()));
    }

    @Test
    public void streamSelection2MockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextTest();

        // A(1,2) -> B(1) -> D(1,3)
        //    \----> C(2)
        // E(4) -> F(4, 5)
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1), new MockArgument(2))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(1))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(2))).schedule();
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1), new MockArgument(3))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(4))).schedule();
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(4), new MockArgument(5))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(2, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(4).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(5).getComputation().getStream().getStreamNumber());
        assertEquals(4, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(0).getComputation().getStream()));
        assertEquals(2, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(4).getComputation().getStream()));
    }

    @Test
    public void streamSelectionSimpleWithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextTest(true);
        // Create 4 mock kernel executions. In this case, kernel 3 requires 1 and 2 to finish,
        //   and kernel 4 requires kernel 3 to finish. The final frontier is composed of kernel 3 (arguments "1" and "2" are active),
        //   and kernel 4 (argument "3" is active);
        // A(1) -> C(1, 2, 3) -> D(3)
        // B(2) /
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(1))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(2))).schedule();
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1),
                        new MockArgument(2),
                        new MockArgument(3))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(3))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(2, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(3).getComputation().getStream()));
        assertEquals(0, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(1).getComputation().getStream()));
    }

    @Test
    public void streamSelectionWithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextTest(true);

        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //     C(2)
        // The final frontier is composed by C(2), D(3), E(1), F(4);
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1), new MockArgument(2))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(1))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(2))).schedule();
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1), new MockArgument(3))).schedule();
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1), new MockArgument(4))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(2, context.getDag().getFrontier().size());
        assertEquals(2, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(4).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(5).getComputation().getStream().getStreamNumber());
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(5).getComputation().getStream()));
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(2).getComputation().getStream()));
    }

    @Test
    public void streamSelection2WithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextTest(true);

        // A(1,2) -> B(1) -> D(1,3)
        //    C(2)
        // E(4) -> F(4, 5)
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1), new MockArgument(2))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(1))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(2))).schedule();
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(1), new MockArgument(3))).schedule();
        new KernelExecutionTest(context, Collections.singletonList(new MockArgument(4))).schedule();
        new KernelExecutionTest(context,
                Arrays.asList(new MockArgument(4), new MockArgument(5))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(3, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(2, dag.getVertices().get(4).getComputation().getStream().getStreamNumber());
        assertEquals(2, dag.getVertices().get(5).getComputation().getStream().getStreamNumber());
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(0).getComputation().getStream()));
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(2).getComputation().getStream()));
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(4).getComputation().getStream()));
    }
}
