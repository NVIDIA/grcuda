package com.nvidia.grcuda.test.gpu.executioncontext;

import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.computation.dependency.WithConstDependencyComputationBuilder;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.test.mock.ArgumentMock;
import com.nvidia.grcuda.test.mock.GrCUDAExecutionContextMock;
import com.nvidia.grcuda.test.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.mock.KernelExecutionMock;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(new WithConstDependencyComputationBuilder());
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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(new WithConstDependencyComputationBuilder());
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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(new WithConstDependencyComputationBuilder());

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
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(new WithConstDependencyComputationBuilder());

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
                .setDependencyComputationBuilder(new WithConstDependencyComputationBuilder()).build();

        // A(1R,2) -> B(1) -> D(1R,3) ---------> G(1,3,4)
        //         \- C(2R) \- E(1R,4) --/--> F(4) -/
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
        assertEquals(8, dag.getNumEdges());
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
    }

    @Test
    public void dependencyPipelineSimpleWithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyComputationBuilder(new WithConstDependencyComputationBuilder()).setSyncStream(true).build();
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
        assertEquals(1, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Collections.singletonList(dag.getVertices().get(3))),
                new HashSet<>(dag.getFrontier()));
        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertTrue(dag.getVertices().get(1).isStart());
        assertFalse(dag.getVertices().get(2).isFrontier());
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
    public void forkedComputationWithSyncTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyComputationBuilder(new WithConstDependencyComputationBuilder()).setSyncStream(true).build();

        // A(1) --> B(1R)
        //          C(1R)
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1, true))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1, true))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(3, dag.getNumVertices());
        assertEquals(1, dag.getNumEdges());
        assertEquals(2, dag.getFrontier().size());

        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertTrue(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertTrue(dag.getVertices().get(2).isStart());

        assertEquals(dag.getVertices().get(0), dag.getVertices().get(1).getParentVertices().get(0));
        assertEquals(0, dag.getVertices().get(2).getParentVertices().size());
        assertFalse(dag.getVertices().get(2).getParentVertices().contains(dag.getVertices().get(1)));
        assertFalse(dag.getVertices().get(1).getChildVertices().contains(dag.getVertices().get(2)));

        // Add a fourth computation that depends on both B and C, and depends on both;
        // A(1) -> B(1R) -> D(1)
        //         C(1R) /
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        assertEquals(4, dag.getNumVertices());
        assertEquals(3, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
    }

    @Test
    public void complexFrontierWithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyComputationBuilder(new WithConstDependencyComputationBuilder()).setSyncStream(true).build();

        // A(1R,2) -> B(1) -> D(1R,3)
        //            C(2R)   E(1R,4) -> F(4)
        // The final frontier is composed by C(2R), D(1R, 3), F(4);
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2, true))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(6, dag.getNumVertices());
        assertEquals(3, dag.getNumEdges());
        assertEquals(3, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(2), dag.getVertices().get(3), dag.getVertices().get(5))),
                new HashSet<>(dag.getFrontier()));

        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertTrue(dag.getVertices().get(2).isStart());
        assertTrue(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        assertFalse(dag.getVertices().get(4).isFrontier());
        assertTrue(dag.getVertices().get(4).isStart());
        assertTrue(dag.getVertices().get(5).isFrontier());
        assertFalse(dag.getVertices().get(5).isStart());
        // Check that D is a child of B and C and D are not connected;
        assertEquals(dag.getVertices().get(1), dag.getVertices().get(3).getParentVertices().get(0));
        assertFalse(dag.getVertices().get(3).getParentVertices().contains(dag.getVertices().get(2)));
        assertFalse(dag.getVertices().get(2).getChildVertices().contains(dag.getVertices().get(3)));
        // Check that D and E are not connected;
        assertFalse(dag.getVertices().get(4).getParentVertices().contains(dag.getVertices().get(3)));
        assertFalse(dag.getVertices().get(3).getChildVertices().contains(dag.getVertices().get(4)));
    }

    @Test
    public void complexFrontier2WithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyComputationBuilder(new WithConstDependencyComputationBuilder()).setSyncStream(true).build();

        // A(1R,2) -> B(1) -> D(1R,3) ---------> G(1, 3, 4)
        //            C(2R)   E(1R,4) -> F(4) -/
        // The final frontier is composed by C(2R), G(1, 3, 4);
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
        assertEquals(5, dag.getNumEdges());
        assertEquals(2, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(2), dag.getVertices().get(6))),
                new HashSet<>(dag.getFrontier()));

        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertTrue(dag.getVertices().get(2).isStart());
        assertFalse(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        assertFalse(dag.getVertices().get(4).isFrontier());
        assertTrue(dag.getVertices().get(4).isStart());
        assertFalse(dag.getVertices().get(5).isFrontier());
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
    }

    private static final int NUM_THREADS_PER_BLOCK = 128;

    private static final String SQUARE_KERNEL =
    "extern \"C\" __global__ void square(const float* x, float *y, int n) {\n" +
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
    "    if (idx < n) {\n" +
    "       y[idx] = x[idx] * x[idx];\n" +
    "    }" +
    "}\n";

    private static final String SQUARE_INPLACE_KERNEL =
    "extern \"C\" __global__ void square(float* x, int n) {\n" +
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
    "    if (idx < n) {\n" +
    "       x[idx] = x[idx] * x[idx];\n" +
    "    }" +
    "}\n";

    private static final String DIFF_KERNEL =
    "extern \"C\" __global__ void diff(const float* x, const float* y, float* z, int n) {\n" +
    "   int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
    "   if (idx < n) {\n" +
    "      z[idx] = x[idx] - y[idx];\n" +
    "   }\n" +
    "}";

    private static final String DIFF_SINGLE_KERNEL =
    "extern \"C\" __global__ void diff(const float* x, float* z, float val, int n) {\n" +
    "   int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
    "   if (idx < n) {\n" +
    "      z[idx] = x[idx] - val;\n" +
    "   }\n" +
    "}";

    private static final String REDUCE_KERNEL =
    "extern \"C\" __global__ void reduce(const float *x, float *res, int n) {\n" +
    "    __shared__ float cache[" + NUM_THREADS_PER_BLOCK + "];\n" +
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
    "    if (i < n) {\n" +
    "       cache[threadIdx.x] = x[i];\n" +
    "    }\n" +
    "    __syncthreads();\n" +
    "    i = " + NUM_THREADS_PER_BLOCK + " / 2;\n" +
    "    while (i > 0) {\n" +
    "       if (threadIdx.x < i) {\n" +
    "            cache[threadIdx.x] += cache[threadIdx.x + i];\n" +
    "        }\n" +
    "        __syncthreads();\n" +
    "        i /= 2;\n" +
    "    }\n" +
    "    if (threadIdx.x == 0) {\n" +
    "        atomicAdd(res, cache[0]);\n" +
    "    }\n" +
    "}";

    @Test
    public void dependencyPipelineSimpleTest() {

        try (Context context = Context.newBuilder().option("grcuda.DependencyPolicy", "with_const").allowAllAccess(true).build()) {

            final int numElements = 10;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "const pointer, pointer, sint32");

            assertNotNull(squareKernel);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, y, numElements);
            configuredSquareKernel.execute(x, z, numElements);

            // Verify the output;
            assertEquals(4.0, y.getArrayElement(0).asFloat(), 0.1);
            assertEquals(4.0, z.getArrayElement(0).asFloat(), 0.1);
            assertEquals(4.0, y.getArrayElement(numElements - 1).asFloat(), 0.1);
            assertEquals(4.0, z.getArrayElement(numElements - 1).asFloat(), 0.1);
        }
    }

    @Test
    public void dependencyPipelineReadXTest() {

        try (Context context = Context.newBuilder().option("grcuda.DependencyPolicy", "with_const").allowAllAccess(true).build()) {

            final int numElements = 100;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "const pointer, pointer, sint32");

            assertNotNull(squareKernel);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, y, numElements);
            configuredSquareKernel.execute(x, z, numElements);

            // Read the array x before syncing the computation. Depending on the GPU, this might sync the device;
            for (int i = 0; i < numElements; ++i) {
                assertEquals(2.0, x.getArrayElement(i).asFloat(), 0.1);
            }

            // Verify the output;
            assertEquals(4.0, y.getArrayElement(0).asFloat(), 0.1);
            assertEquals(4.0, z.getArrayElement(0).asFloat(), 0.1);
            assertEquals(4.0, y.getArrayElement(numElements - 1).asFloat(), 0.1);
            assertEquals(4.0, z.getArrayElement(numElements - 1).asFloat(), 0.1);
        }
    }

    @Test
    public void dependencyPipelineSplitComputationTest() {
        // Test a computation of form A(1) --> B(1r, 2)
        //                                 \-> C(1r, 3)
        try (Context context = Context.newBuilder().option("grcuda.DependencyPolicy", "with_const").allowAllAccess(true).build()) {

            final int numElements = 10;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_INPLACE_KERNEL, "square", "pointer, sint32");
            Value diffKernel = buildkernel.execute(DIFF_SINGLE_KERNEL, "diff", "const pointer, pointer, float, sint32");

            assertNotNull(squareKernel);
            assertNotNull(diffKernel);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredDiffKernel = diffKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, numElements);
            configuredDiffKernel.execute(x, y, 1.0, numElements);
            configuredDiffKernel.execute(x, z, 1.0, numElements);

            // Verify the output;
            assertEquals(3.0, y.getArrayElement(0).asFloat(), 0.1);
            assertEquals(3.0, z.getArrayElement(0).asFloat(), 0.1);
            assertEquals(3.0, y.getArrayElement(numElements - 1).asFloat(), 0.1);
            assertEquals(3.0, z.getArrayElement(numElements - 1).asFloat(), 0.1);
            assertEquals(4.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(4.0, x.getArrayElement(0).asFloat(), 0.1);
        }
    }

    @Test
    public void dependencyPipelineDiamondTest() {
        // Test a computation of form A(1) --> B(1r, 2) -> D(1)
        //                                 \-> C(1r, 3) -/
        try (Context context = Context.newBuilder().option("grcuda.DependencyPolicy", "with_const").allowAllAccess(true).build()) {

            final int numElements = 10;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_INPLACE_KERNEL, "square", "pointer, sint32");
            Value diffKernel = buildkernel.execute(DIFF_SINGLE_KERNEL, "diff", "const pointer, pointer, float, sint32");

            assertNotNull(squareKernel);
            assertNotNull(diffKernel);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredDiffKernel = diffKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, numElements);
            configuredDiffKernel.execute(x, y, 1.0, numElements);
            configuredDiffKernel.execute(x, z, 1.0, numElements);
            configuredSquareKernel.execute(x, numElements);

            // Verify the output;
            assertEquals(16.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(16.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(3.0, y.getArrayElement(0).asFloat(), 0.1);
            assertEquals(3.0, z.getArrayElement(0).asFloat(), 0.1);
            assertEquals(3.0, y.getArrayElement(numElements - 1).asFloat(), 0.1);
            assertEquals(3.0, z.getArrayElement(numElements - 1).asFloat(), 0.1);
        }
    }


    @Test
    public void dependencyPipelineSimple2Test() {

        try (Context context = Context.newBuilder().option("grcuda.DependencyPolicy", "with_const").allowAllAccess(true).build()) {

            final int numElements = 100000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);
            Value w = deviceArrayConstructor.execute("float", numElements);
            Value res = deviceArrayConstructor.execute("float", 1);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 1.0 / (i + 1));
            }
            res.setArrayElement(0, 0.0);

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "const pointer, pointer, sint32");
            Value diffKernel = buildkernel.execute(DIFF_KERNEL, "diff", "const pointer, const pointer, pointer, sint32");
            Value reduceKernel = buildkernel.execute(REDUCE_KERNEL, "reduce", "const pointer, pointer, sint32");
            assertNotNull(squareKernel);
            assertNotNull(diffKernel);
            assertNotNull(reduceKernel);

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredDiffKernel = diffKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredReduceKernel = reduceKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, y, numElements);
            configuredSquareKernel.execute(x, z, numElements);
            configuredDiffKernel.execute(y, z, w, numElements);
            configuredReduceKernel.execute(w, res, numElements);

            // Verify the output;
            float resScalar = res.getArrayElement(0).asFloat();
            assertEquals(0, resScalar, 0.01);
        }
    }
}
