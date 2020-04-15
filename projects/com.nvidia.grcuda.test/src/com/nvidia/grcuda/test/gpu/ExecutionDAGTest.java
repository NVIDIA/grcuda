package com.nvidia.grcuda.test.gpu;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDALanguage;
import com.nvidia.grcuda.array.DeviceArray;
import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.InitializeArgumentSet;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.instrumentation.EventContext;
import com.oracle.truffle.api.instrumentation.ExecutionEventListener;
import com.oracle.truffle.api.instrumentation.LoadSourceEvent;
import com.oracle.truffle.api.instrumentation.LoadSourceListener;
import com.oracle.truffle.api.instrumentation.LoadSourceSectionEvent;
import com.oracle.truffle.api.instrumentation.SourceFilter;
import com.oracle.truffle.api.instrumentation.SourceSectionFilter;
import com.oracle.truffle.api.instrumentation.StandardTags;
import com.oracle.truffle.api.instrumentation.TruffleInstrument;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.source.Source;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Engine;
import org.graalvm.polyglot.HostAccess;
import org.graalvm.polyglot.Instrument;
import org.graalvm.polyglot.Value;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class ExecutionDAGTest {

    /**
     * Mock class to test the DAG execution;
     */
    private static class KernelExecutionTest extends GrCUDAComputationalElement {
        KernelExecutionTest(List<Object> args) {
            super(new KernelExecutionTestInitializer(args));
        }
    }
    /**
     * Mock class to test KernelExecutionTest initialization;
     */
    private static class KernelExecutionTestInitializer implements InitializeArgumentSet {
        List<Object> args;
        KernelExecutionTestInitializer(List<Object> args) {
            this.args = args;
        }
        @Override
        public Set<Object> initialize() {
            return new HashSet<>(args);
        }
    }

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
    public void addVertexToDAGTest() {
        ExecutionDAG dag = new ExecutionDAG();
        // Create two mock kernel executions;
        KernelExecutionTest kernel1 = new KernelExecutionTest(Arrays.asList(1, 2, 3));
        KernelExecutionTest kernel2 = new KernelExecutionTest(Arrays.asList(1, 2, 3));

        dag.append(kernel1);

        assertEquals(1, dag.getNumVertices());
        assertEquals(0, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
        assertTrue(dag.getFrontier().get(0).isFrontier());
        assertTrue(dag.getFrontier().get(0).isStart());

        dag.append(kernel2);

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
        assertEquals(dag.getVertices().get(0), dag.getVertices().get(1).getParents().get(0).getStart());
        // Check if the second vertex is a child of the first;
        assertEquals(dag.getVertices().get(1), dag.getVertices().get(0).getChildren().get(0).getEnd());
    }

    @Test
    public void dependencyPipelineSimpleMockTest() {
        ExecutionDAG dag = new ExecutionDAG();
        // Create 4 mock kernel executions;
        KernelExecutionTest kernel1 = new KernelExecutionTest(Collections.singletonList(1));
        KernelExecutionTest kernel2 = new KernelExecutionTest(Collections.singletonList(2));
        KernelExecutionTest kernel3 = new KernelExecutionTest(Arrays.asList(1, 2, 3));
        KernelExecutionTest kernel4 = new KernelExecutionTest(Collections.singletonList(3));

        dag.append(kernel1);
        dag.append(kernel2);
        dag.append(kernel3);
        dag.append(kernel4);

        // Check the DAG structure;
        assertEquals(4, dag.getNumVertices());
        assertEquals(3, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(dag.getVertices().get(3), dag.getFrontier().get(0));
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
        assertEquals(dag.getVertices().get(2).getParents().stream()
                .map(ExecutionDAG.DAGEdge::getStart)
                .collect(Collectors.toSet()),
                new HashSet<>(Arrays.asList(dag.getVertices().get(0), dag.getVertices().get(1))));
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(0).getChildren().get(0).getEnd());
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(1).getChildren().get(0).getEnd());
        // Check if the fourth vertex is a child of the third;
        assertEquals(1, dag.getVertices().get(3).getParents().size());
        assertEquals(1, dag.getVertices().get(2).getChildren().size());
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(3).getParents().get(0).getStart());
        assertEquals(dag.getVertices().get(3), dag.getVertices().get(2).getChildren().get(0).getEnd());
    }

    private static final int NUM_THREADS_PER_BLOCK = 128;

    private static final String SQUARE_KERNEL =
    "extern \"C\" __global__ void square(float* x, int n) {\n" +
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
    "    if (idx < n) {\n" +
    "       x[idx] = x[idx] * x[idx];\n" +
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

            // TODO: is there a way to access the inner GrCUDA data structures?

            // FIXME: the computation gives a wrong numerical value for small N (< 100000), but only in Java (not in Graalpython),
            //   and without any change to the runtime.
            final int numElements = 100000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);
            Value res = deviceArrayConstructor.execute("float", 1);

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            Value diffKernel = buildkernel.execute(DIFF_KERNEL, "diff", "pointer, pointer, pointer, sint32");
            Value reduceKernel = buildkernel.execute(REDUCE_KERNEL, "reduce", "pointer, pointer, sint32");
            assertNotNull(squareKernel);
            assertNotNull(diffKernel);
            assertNotNull(reduceKernel);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 1.0 / (i + 1));
                y.setArrayElement(i, 2.0 / (i + 1));
                z.setArrayElement(i, 0.0);
            }
            res.setArrayElement(0, 0);

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
}
