package com.nvidia.grcuda.test.gpu;

import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.test.mock.GrCUDAExecutionContextMock;
import com.nvidia.grcuda.test.mock.KernelExecutionMock;
import com.nvidia.grcuda.test.mock.ArgumentMock;
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

public class ExecutionDAGTest {

    @Test
    public void executionDAGConstructorTest() {
        ExecutionDAG dag = new ExecutionDAG();
        assertTrue(dag.getVertices().isEmpty());
        assertTrue(dag.getEdges().isEmpty());
        assertTrue(dag.getFrontier().isEmpty());
        assertEquals(0, dag.getNumVertices());
        assertEquals(0, dag.getNumEdges());
    }

    @Test
    public void addVertexToDAGTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock();
        // Create two mock kernel executions;
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2), new ArgumentMock(3))).schedule();

        ExecutionDAG dag = context.getDag();

        assertEquals(1, dag.getNumVertices());
        assertEquals(0, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
        assertTrue(dag.getFrontier().get(0).isFrontier());
        assertTrue(dag.getFrontier().get(0).isStart());

        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2), new ArgumentMock(3))).schedule();

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
    }

    @Test
    public void dependencyPipelineSimpleMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock();
        // Create 4 mock kernel executions. In this case, kernel 3 requires 1 and 2 to finish,
        //   and kernel 4 requires kernel 3 to finish. The final frontier is composed of kernel 3 (arguments "1" and "2" are active),
        //   and kernel 4 (argument "3" is active);
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(3))).schedule();

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
    public void complexFrontierMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock();

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

        // Check the DAG structure;
        assertEquals(6, dag.getNumVertices());
        assertEquals(5, dag.getNumEdges());
        assertEquals(4, dag.getFrontier().size());
        // Check updates to frontier and start status;
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
    }

    @Test
    public void complexFrontierWithSyncMockTest() throws UnsupportedTypeException {
        GrCUDAExecutionContext context = new GrCUDAExecutionContextMock(true);

        // This time, simulate the synchronization process between kernels;
        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //    C(2)
        // The final frontier is composed by C(2) F(4);
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(6, dag.getNumVertices());
        assertEquals(4, dag.getNumEdges());
        assertEquals(2, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(2), dag.getVertices().get(5))),
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
        assertFalse(dag.getVertices().get(4).isStart());
        assertTrue(dag.getVertices().get(5).isFrontier());
        assertFalse(dag.getVertices().get(5).isStart());
    }

    private static final int NUM_THREADS_PER_BLOCK = 128;

    private static final String SQUARE_KERNEL =
    "extern \"C\" __global__ void square(float* x, int n) {\n" +
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
    "    if (idx < n) {\n" +
    "       x[idx] = x[idx] * x[idx];\n" +
    "    }" +
    "}\n";

    private static final String SQUARE_2_KERNEL =
    "extern \"C\" __global__ void square(const float* x, float *y, int n) {\n" +
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
    "    if (idx < n) {\n" +
    "       y[idx] = x[idx] * x[idx];\n" +
    "    }" +
    "}\n";

    private static final String DIFF_KERNEL =
    "extern \"C\" __global__ void diff(float* x, float* y, float* z, int n) {\n" +
    "   int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
    "   if (idx < n) {\n" +
    "      z[idx] = x[idx] - y[idx];\n" +
    "   }\n" +
    "}";

    private static final String REDUCE_KERNEL =
    "extern \"C\" __global__ void reduce(float *x, float *res, int n) {\n" +
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

        try (Context context = Context.newBuilder().allowAllAccess(true).build()) {

            final int numElements = 10;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");

            assertNotNull(squareKernel);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
                y.setArrayElement(i, 4.0);
            }

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, numElements);
            configuredSquareKernel.execute(y, numElements);

            // Verify the output;
            assertEquals(4.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(16.0, y.getArrayElement(0).asFloat(), 0.1);
        }
    }

    @Test
    public void dependencyPipelineSimple2Test() {

        try (Context context = Context.newBuilder().allowAllAccess(true).build()) {

            // FIXME: this test fails randomly with small values (< 100000, more or less),
            //  but the same computation doesn't fail in Graalpython.
            final int numElements = 100000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);
            Value res = deviceArrayConstructor.execute("float", 1);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 1.0 / (i + 1));
                y.setArrayElement(i, 2.0 / (i + 1));
            }
            res.setArrayElement(0, 0.0);

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            Value diffKernel = buildkernel.execute(DIFF_KERNEL, "diff", "const pointer, const pointer, pointer, sint32");
            Value reduceKernel = buildkernel.execute(REDUCE_KERNEL, "reduce", "const pointer, pointer, sint32");
            assertNotNull(squareKernel);
            assertNotNull(diffKernel);
            assertNotNull(reduceKernel);

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredDiffKernel = diffKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredReduceKernel = reduceKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, numElements);
            configuredSquareKernel.execute(y, numElements);
            configuredDiffKernel.execute(x, y, z, numElements);
            configuredReduceKernel.execute(z, res, numElements);

            // Verify the output;
            float resScalar = res.getArrayElement(0).asFloat();
            assertEquals(-4.93, resScalar, 0.01);
        }
    }

    /**
     * The read on "y" has to sync on the stream where the kernel is running, although that kernel doesn't use "y".
     * This is due to the pre-Pascal limitations on managed memory accesses,
     * and the inability to access an array while it is being used by a running kernel;
     */
    @Test
    public void dependencyPipelineSimple3Test() {

        try (Context context = Context.newBuilder().allowAllAccess(true).build()) {

            final int numElements = 100000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_2_KERNEL, "square", "const pointer, pointer, sint32");
            assertNotNull(squareKernel);

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, z, numElements);

            // Verify the output;
            assertEquals(0.0, y.getArrayElement(0).asFloat(), 0.1);
            assertEquals(4.0, z.getArrayElement(0).asFloat(), 0.1);
        }
    }

    @Test
    public void dependencyPipelineSimple4Test() {

        try (Context context = Context.newBuilder().allowAllAccess(true).build()) {

            final int numElements = 100000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }
            // FIXME: if I write on array x, launch K(Y) then read(x), the last comp on x is array access, so no sync is done!!!
            y.setArrayElement(0, 0);

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_2_KERNEL, "square", "const pointer, pointer, sint32");
            assertNotNull(squareKernel);

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, z, numElements);

            // Verify the output;
            assertEquals(0.0, y.getArrayElement(0).asFloat(), 0.1);
            assertEquals(4.0, z.getArrayElement(0).asFloat(), 0.1);
        }
    }
}
