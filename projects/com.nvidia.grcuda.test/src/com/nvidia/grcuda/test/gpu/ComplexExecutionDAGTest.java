package com.nvidia.grcuda.test.gpu;

import com.nvidia.grcuda.gpu.computation.KernelExecution;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.GrCUDATestUtil;
import com.nvidia.grcuda.test.mock.ArgumentMock;
import com.nvidia.grcuda.test.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.mock.GrCUDAStreamManagerMock;
import com.nvidia.grcuda.test.mock.KernelExecutionMock;
import com.nvidia.grcuda.test.mock.SyncExecutionMock;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(Parameterized.class)
public class ComplexExecutionDAGTest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {

        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {RetrieveNewStreamPolicyEnum.ALWAYS_NEW, RetrieveNewStreamPolicyEnum.FIFO},
                {RetrieveParentStreamPolicyEnum.DISJOINT, RetrieveParentStreamPolicyEnum.DEFAULT},
                {DependencyPolicyEnum.WITH_CONST, DependencyPolicyEnum.DEFAULT}
        }));
    }

    private final RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy;
    private final RetrieveParentStreamPolicyEnum retrieveParentStreamPolicy;
    private final DependencyPolicyEnum dependencyPolicy;

    public ComplexExecutionDAGTest(RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy,
                                   RetrieveParentStreamPolicyEnum retrieveParentStreamPolicy,
                                   DependencyPolicyEnum dependencyPolicy) {
        this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
        this.retrieveParentStreamPolicy = retrieveParentStreamPolicy;
        this.dependencyPolicy = dependencyPolicy;
    }

    @Test
    public void hitsMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy).setRetrieveParentStreamPolicy(this.retrieveParentStreamPolicy)
                .setDependencyPolicy(this.dependencyPolicy).build();

        int numIterations = 10;
        KernelExecutionMock c1 = null;
        KernelExecutionMock c2 = null;
        for (int i = 0; i < numIterations; i++) {
            // hub1 -> auth2
            c1 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(2)));
            c1.schedule();
            // auth1 -> hub2
            c2 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(3, true), new ArgumentMock(4)));
            c2.schedule();

            // Without disjoint policy the computation collapses on a single stream after the first iteration;
            int stream = (retrieveParentStreamPolicy.equals(RetrieveParentStreamPolicyEnum.DISJOINT) || i == 0) ? 0 : 1;
            assertEquals(stream, c1.getStream().getStreamNumber());
            assertEquals(1, c2.getStream().getStreamNumber());

            // auth2 -> auth_norm
            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(2, true), new ArgumentMock(5))).schedule();
            // hub2 -> hub_norm
            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(4, true), new ArgumentMock(6))).schedule();
            // auth2, auth_norm -> auth1
            c1 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(2, true), new ArgumentMock(5, true), new ArgumentMock(3)));
            c1.schedule();
            // hub2, hub_norm -> hub1
            c2 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(4, true), new ArgumentMock(6, true), new ArgumentMock(1)));
            c2.schedule();
        }

        assertEquals(2, context.getStreamManager().getNumberOfStreams());

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(3))).schedule();
        assertTrue(context.getStreamManager().isStreamFree(c1.getStream()));
        int activeComps = retrieveParentStreamPolicy.equals(RetrieveParentStreamPolicyEnum.DISJOINT) ? 2 : 0;
        assertEquals(activeComps, context.getStreamManager().getNumActiveComputationsOnStream(c2.getStream()));

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        assertTrue(context.getStreamManager().isStreamFree(c1.getStream()));
        assertTrue(context.getStreamManager().isStreamFree(c2.getStream()));
    }

    @Test
    public void imageMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy).setRetrieveParentStreamPolicy(this.retrieveParentStreamPolicy)
                .setDependencyPolicy(this.dependencyPolicy).build();

        // blur
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(4))).schedule();
        // sobel
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(2, true), new ArgumentMock(5))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(3, true), new ArgumentMock(6))).schedule();
        // extend
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(4, true), new ArgumentMock(7))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(4, true), new ArgumentMock(8))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(4), new ArgumentMock(7), new ArgumentMock(8))).schedule();
        // unsharpen
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(4), new ArgumentMock(9))).schedule();
        // combine
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(9, true), new ArgumentMock(3, true),
                new ArgumentMock(6, true), new ArgumentMock(10))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(10, true), new ArgumentMock(2, true),
                new ArgumentMock(5, true), new ArgumentMock(11))).schedule();

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(11))).schedule();
        int numStreams = 3;
        if (retrieveParentStreamPolicy.equals(RetrieveParentStreamPolicyEnum.DISJOINT) && dependencyPolicy.equals(DependencyPolicyEnum.WITH_CONST)) {
            numStreams = 4;
        }
        else if (retrieveParentStreamPolicy.equals(RetrieveParentStreamPolicyEnum.DEFAULT) && dependencyPolicy.equals(DependencyPolicyEnum.DEFAULT)) {
            numStreams = 1;
        }
        assertEquals(numStreams, context.getStreamManager().getNumberOfStreams());
        for (CUDAStream stream : ((GrCUDAStreamManagerMock) context.getStreamManager()).getStreams()) {
            assertTrue(context.getStreamManager().isStreamFree(stream));
        }
    }
}
