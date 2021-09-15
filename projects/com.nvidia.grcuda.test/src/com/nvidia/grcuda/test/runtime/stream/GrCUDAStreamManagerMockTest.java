package com.nvidia.grcuda.test.runtime.stream;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.mock.ArgumentMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.util.mock.GrCUDAStreamManagerMock;
import com.nvidia.grcuda.test.util.mock.KernelExecutionMock;
import com.nvidia.grcuda.test.util.mock.SyncExecutionMock;
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
public class GrCUDAStreamManagerMockTest {
    /**
     * Tests are executed for each of the {@link RetrieveNewStreamPolicyEnum} values;
     *
     * @return the current stream policy
     */
    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
                {RetrieveNewStreamPolicyEnum.ALWAYS_NEW},
                {RetrieveNewStreamPolicyEnum.FIFO},
        });
    }

    private final RetrieveNewStreamPolicyEnum policy;

    public GrCUDAStreamManagerMockTest(RetrieveNewStreamPolicyEnum policy) {
        this.policy = policy;
    }

    @Test
    public void streamSelectionSimpleMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveNewStreamPolicy(this.policy).build();
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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveNewStreamPolicy(this.policy).build();

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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveNewStreamPolicy(this.policy).build();

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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveNewStreamPolicy(this.policy).build();
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
        assertEquals(3, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(3).getComputation().getStream()));
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(1).getComputation().getStream()));

        // Synchronize computations;
        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(3))).schedule();
        // The stream has no active computation;
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(1).getComputation().getStream()));
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(3).getComputation().getStream()));
    }

    @Test
    public void streamSelectionWithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveNewStreamPolicy(this.policy).build();

        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //    \-> C(2)
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
        assertEquals(4, context.getDag().getFrontier().size());
        // In this simple test, do not use disjoint stream assignment;
        assertEquals(1, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(4).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(5).getComputation().getStream().getStreamNumber());
        assertEquals(6, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(5).getComputation().getStream()));

        // All computations are on the same stream, so syncing one will terminate all of them;
        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(5).getComputation().getStream()));
    }

    @Test
    public void streamSelection2WithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveNewStreamPolicy(this.policy).build();

        // A(1,2) -> B(1) -> D(1,3)
        //   \-> C(2)
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

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(2).getComputation().getStream()));
        assertEquals(2, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(4).getComputation().getStream()));

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(5).getComputation().getStream()));
    }

    @Test
    public void generateManyStreamsTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder().setRetrieveNewStreamPolicy(this.policy).build();

        // Create 2 parallel branches on dependent computations, and check that the total amount of streams created is what is expected;
        int numLoops = 10;
        for (int i = 0; i < numLoops * 2; i += 2) {
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(i))).schedule();
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(i + 1))).schedule();
            // Sync point;
            new SyncExecutionMock(context, Arrays.asList(new ArgumentMock(i), new ArgumentMock(i + 1))).schedule();
        }

        ExecutionDAG dag = context.getDag();
        // Check that kernels have been given the right stream;
        int numStreams = this.policy == RetrieveNewStreamPolicyEnum.FIFO ? 2 : numLoops * 2;
        int streamCheck1 = this.policy == RetrieveNewStreamPolicyEnum.FIFO ? 0 : numLoops * 2 - 2;
        int streamCheck2 = this.policy == RetrieveNewStreamPolicyEnum.FIFO ? 1 : numLoops * 2 - 1;

        assertEquals(numStreams, context.getStreamManager().getNumberOfStreams());
        assertEquals(streamCheck1, dag.getVertices().get(numLoops * 3 - 3).getComputation().getStream().getStreamNumber());
        assertEquals(streamCheck2, dag.getVertices().get(numLoops * 3 - 2).getComputation().getStream().getStreamNumber());
    }

    @Test
    public void disjointArgumentStreamTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.policy).setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT).build();

        // A(1,2) -> B(1)
        //   \-> C(2)
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(2, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(2, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(1).getComputation().getStream()));
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(2).getComputation().getStream()));

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(2).getComputation().getStream()));
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(0).getComputation().getStream()));
        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(0).getComputation().getStream()));
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(1).getComputation().getStream()));
    }

    @Test
    public void disjointArgumentStreamCrossTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.policy).setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT).build();

        // A(1,2) -> C(1,3)
        //        X
        // B(3,4) -> D(2,4)
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(3), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(2), new ArgumentMock(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check that kernels have been given the right stream;
        assertEquals(2, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(2, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(2).getComputation().getStream()));
        assertEquals(2, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(3).getComputation().getStream()));
    }

    @Test
    public void disjointArgumentStreamCross2Test() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
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

        // Check that kernels have been given the right stream;
        assertEquals(3, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(2, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(4).getComputation().getStream().getStreamNumber());
        assertEquals(2, dag.getVertices().get(5).getComputation().getStream().getStreamNumber());
        assertEquals(2, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(3).getComputation().getStream()));
        assertEquals(2, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(4).getComputation().getStream()));
        assertEquals(2, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(5).getComputation().getStream()));
    }

    @Test
    public void syncParentsOfParentsTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
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

        // Check that kernels have been given the right stream;
        assertEquals(3, context.getStreamManager().getNumberOfStreams());
        assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        assertEquals(0, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(2).getComputation().getStream().getStreamNumber());
        assertEquals(1, dag.getVertices().get(3).getComputation().getStream().getStreamNumber());
        assertEquals(2, dag.getVertices().get(4).getComputation().getStream().getStreamNumber());
        assertEquals(2, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(0).getComputation().getStream()));
        assertEquals(2, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(2).getComputation().getStream()));
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(4).getComputation().getStream()));

        // Syncing E(3) will sync also computations on stream 1 and 0;
        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(3))).schedule();
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(3).getComputation().getStream()));
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(1).getComputation().getStream()));
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(4).getComputation().getStream()));
        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        assertEquals(1, context.getStreamManager().getNumActiveComputationsOnStream(dag.getVertices().get(3).getComputation().getStream()));
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(1).getComputation().getStream()));

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        assertFalse(((GrCUDAStreamManagerMock) context.getStreamManager()).getActiveComputationsMap().containsKey(dag.getVertices().get(3).getComputation().getStream()));
    }

    @Test
    public void repeatedSyncTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.policy).build();

        int numTest = 10;
        ExecutionDAG dag = context.getDag();

        for (int i = 0; i < numTest; i++) {
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
            new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();

            new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
            new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
            assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
            assertEquals(1, dag.getVertices().get(1).getComputation().getStream().getStreamNumber());
            assertEquals(0, dag.getVertices().get(0).getComputation().getStream().getStreamNumber());
        }
    }
}
