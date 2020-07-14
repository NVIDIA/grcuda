package com.nvidia.grcuda.test.gpu.executioncontext;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class GrCUDAExecutionContextWithConstDependencyTest {

    private static final int NUM_THREADS_PER_BLOCK = 32;

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

        try (Context context = Context.newBuilder().option("grcuda.DependencyPolicy", "with-const").allowAllAccess(true).build()) {

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

        try (Context context = Context.newBuilder().option("grcuda.DependencyPolicy", "with-const").allowAllAccess(true).build()) {

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
        try (Context context = Context.newBuilder().option("grcuda.DependencyPolicy", "with-const").allowAllAccess(true).build()) {

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
        try (Context context = Context.newBuilder().option("grcuda.DependencyPolicy", "with-const").allowAllAccess(true).build()) {

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

        try (Context context = Context.newBuilder().option("grcuda.DependencyPolicy", "with-const").allowAllAccess(true).build()) {

            final int numElements = 10000;
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
