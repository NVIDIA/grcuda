package com.nvidia.grcuda.test.gpu.stream;

import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.RetrieveStreamPolicyEnum;
import com.nvidia.grcuda.test.mock.ArgumentMock;
import com.nvidia.grcuda.test.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.mock.GrCUDAStreamManagerMock;
import com.nvidia.grcuda.test.mock.KernelExecutionMock;
import com.nvidia.grcuda.test.mock.SyncExecutionMock;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

@RunWith(Parameterized.class)
public class GrCUDAStreamManagerTest {
    /**
     * Tests are executed for each of the {@link com.nvidia.grcuda.gpu.stream.RetrieveStreamPolicyEnum} values;
     * @return the current stream policy
     */
    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
                {RetrieveStreamPolicyEnum.ALWAYS_NEW},
                {RetrieveStreamPolicyEnum.LIFO},
        });
    }

    private final RetrieveStreamPolicyEnum policy;

    public GrCUDAStreamManagerTest(RetrieveStreamPolicyEnum policy) {
        this.policy = policy;
    }

    @Test
    public void streamSelectionSimpleMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveStreamPolicy(this.policy).build();
        // Create 4 mock kernel executions. In this case, kernel 3 requires 1 and 2 to finish,
        //   and kernel 4 requires kernel 3 to finish. The final frontier is composed of kernel 3 (arguments "1" and "2" are active),
        //   and kernel 4 (argument "3" is active);
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1),
                        new ArgumentMock(2),
                        new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(3))).schedule();

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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveStreamPolicy(this.policy).build();

        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //    \----> C(2)
        // The final frontier is composed by C(2), D(3), E(1), F(4);
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();

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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveStreamPolicy(this.policy).build();

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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setSyncStream(true).setRetrieveStreamPolicy(this.policy).build();
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

        // Check that kernels have been given the right stream;
        assertEquals(2, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(3).getComputation().getStream()));
        // The stream has no active computation;
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(1).getComputation().getStream()));
    }

    @Test
    public void streamSelectionWithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setSyncStream(true).setRetrieveStreamPolicy(this.policy).build();

        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //     C(2)
        // The final frontier is composed by C(2), D(3), E(1), F(4);
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();

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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setSyncStream(true).setRetrieveStreamPolicy(this.policy).build();

        // A(1,2) -> B(1) -> D(1,3)
        //    C(2)
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

    @Test
    public void generateManyStreamsTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setSyncStream(true).setRetrieveStreamPolicy(this.policy).build();

        // Create 2 parallel branches on dependent computations, and check that the total amount of streams created is what is expected;
        int numLoops = 10;
        for (int i = 0; i < numLoops * 2; i+=2) {
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(i))).schedule();
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(i + 1))).schedule();
            // Sync point;
            new SyncExecutionMock(context, Arrays.asList(new ArgumentMock(i), new ArgumentMock(i + 1))).schedule();
        }

        ExecutionDAG dag = context.getDag();
        // Check that kernels have been given the right stream;
        int numStreams = this.policy == RetrieveStreamPolicyEnum.LIFO ? 2 : numLoops * 2;
        int streamCheck1 = this.policy == RetrieveStreamPolicyEnum.LIFO ? 0 : numLoops * 2 - 2;
        int streamCheck2 = this.policy == RetrieveStreamPolicyEnum.LIFO ? 1 : numLoops * 2 - 1;

        assertEquals(numStreams, context.getStreamManager().getNumberOfStreams());
        assertEquals(streamCheck1, dag.getVertices().get(numLoops * 3 - 3).getComputation().getStream().getStreamNumber());
        assertEquals(streamCheck2, dag.getVertices().get(numLoops * 3 - 2).getComputation().getStream().getStreamNumber());
    }
}
