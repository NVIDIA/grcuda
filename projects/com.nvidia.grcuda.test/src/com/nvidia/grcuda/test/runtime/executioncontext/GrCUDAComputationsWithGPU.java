package com.nvidia.grcuda.test.runtime.executioncontext;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Class that contains static functions that are used as building blocks for GrCUDA tests that use the GPU;
 */
public class GrCUDAComputationsWithGPU {

    static final int NUM_THREADS_PER_BLOCK = 32;

    static final String SQUARE_WITH_CONST =
            "extern \"C\" __global__ void square(const float* x, float *y, int n) {\n" +
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (idx < n) {\n" +
                    "       y[idx] = x[idx] * x[idx];\n" +
                    "    }" +
                    "}\n";

    static final String DIFF_KERNEL =
            "extern \"C\" __global__ void diff(float* x, float* y, float* z, int n) {\n" +
                    "   int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "   if (idx < n) {\n" +
                    "      z[idx] = x[idx] - y[idx];\n" +
                    "   }\n" +
                    "}";

    static final String REDUCE_KERNEL =
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

    static final String SQUARE_INPLACE_KERNEL =
            "extern \"C\" __global__ void square(float* x, int n) {\n" +
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (idx < n) {\n" +
                    "       x[idx] = x[idx] * x[idx];\n" +
                    "    }" +
                    "}\n";

    static final String DIFF_SINGLE_KERNEL =
            "extern \"C\" __global__ void diff(const float* x, float* z, float val, int n) {\n" +
                    "   int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "   if (idx < n) {\n" +
                    "      z[idx] = x[idx] - val;\n" +
                    "   }\n" +
                    "}";

    /**
     * Test a simple join pattern with the following shape;
     * (X) --> (Z)
     * (Y) -/
     * @param context context a GrCUDA context with the specified options
     */
    static void simpleJoin(Context context) {
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
        Value squareKernel = buildkernel.execute(SQUARE_INPLACE_KERNEL, "square", "pointer, sint32");
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

    /**
     * Test a simple join pattern with the following shape. In this case, also test the copy functions by moving data
     * from X, Y into X2, Y2;
     * (X) -copy-> (X2) --> (Z)
     * (Y) -copy-> (Y2) -/
     * @param context a GrCUDA context with the specified options
     */
    static void arrayCopyWithJoin(Context context) {
        final int numElements = 100000;
        final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
        Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
        Value x = deviceArrayConstructor.execute("float", numElements);
        Value y = deviceArrayConstructor.execute("float", numElements);
        Value z = deviceArrayConstructor.execute("float", numElements);
        Value x2 = deviceArrayConstructor.execute("float", numElements);
        Value y2 = deviceArrayConstructor.execute("float", numElements);
        Value res = deviceArrayConstructor.execute("float", 1);
        Value res2 = deviceArrayConstructor.execute("float", 1);

        for (int i = 0; i < numElements; ++i) {
            x.setArrayElement(i, 1.0 / (i + 1));
            y.setArrayElement(i, 2.0 / (i + 1));
        }
        res.setArrayElement(0, 0.0);

        x2.invokeMember("copyFrom", x, numElements);
        y2.invokeMember("copyFrom", y, numElements);

        Value buildkernel = context.eval("grcuda", "buildkernel");
        Value squareKernel = buildkernel.execute(SQUARE_INPLACE_KERNEL, "square", "pointer, sint32");
        Value diffKernel = buildkernel.execute(DIFF_KERNEL, "diff", "const pointer, const pointer, pointer, sint32");
        Value reduceKernel = buildkernel.execute(REDUCE_KERNEL, "reduce", "const pointer, pointer, sint32");
        assertNotNull(squareKernel);
        assertNotNull(diffKernel);
        assertNotNull(reduceKernel);

        Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
        Value configuredDiffKernel = diffKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
        Value configuredReduceKernel = reduceKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

        // Perform the computation;
        configuredSquareKernel.execute(x2, numElements);
        configuredSquareKernel.execute(y2, numElements);
        configuredDiffKernel.execute(x2, y2, z, numElements);
        configuredReduceKernel.execute(z, res, numElements);

        res.invokeMember("copyTo", res2, 1);

        // Verify the output;
        float resScalar = res2.getArrayElement(0).asFloat();
        assertEquals(-4.93, resScalar, 0.01);
    }

    /**
     * Execute two kernels with read only arguments;
     * (X) --> (Y)
     *     \-> (Z)
     * @param context a GrCUDA context with the specified options
     */
    static void parallelKernelsWithReadOnlyArgs(Context context) {
        final int numElements = 10;
        final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
        Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
        Value x = deviceArrayConstructor.execute("float", numElements);
        Value y = deviceArrayConstructor.execute("float", numElements);
        Value z = deviceArrayConstructor.execute("float", numElements);
        Value buildkernel = context.eval("grcuda", "buildkernel");
        Value squareKernel = buildkernel.execute(SQUARE_WITH_CONST, "square", "const pointer, pointer, sint32");

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

    /**
     * Execute a fork, with read only arguments.
     * Read the input array x before syncing the computation. Depending on the GPU, this might sync the device;
     * (X) --> (Y)
     *     \-> (Z)
     * @param context a GrCUDA context with the specified options
     */
    static void simpleForkReadInput(Context context) {
        final int numElements = 100;
        final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
        Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
        Value x = deviceArrayConstructor.execute("float", numElements);
        Value y = deviceArrayConstructor.execute("float", numElements);
        Value z = deviceArrayConstructor.execute("float", numElements);
        Value buildkernel = context.eval("grcuda", "buildkernel");
        Value squareKernel = buildkernel.execute(SQUARE_WITH_CONST, "square", "const pointer, pointer, sint32");

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

    /**
     * Execute a fork, with a read only argument in the children kernels;
     * A(1) --> B(1r, 2)
     *      \-> C(1r, 3)
     * @param context a GrCUDA context with the specified options
     */
    static void forkWithReadOnly(Context context) {
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

    /**
     * Execute a diamond, i.e. a fork with read-only arguments in the children followed by a join;
     * A(1) --> B(1r, 2) -> D(1)
     *      \-> C(1r, 3) -/
     * @param context a GrCUDA context with the specified options
     */
    static void dependencyPipelineDiamond(Context context) {
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

    /**
     * Execute a join followed by an extra kernel, using read-only arguments;
     * (X) --> (Z) -> (R)
     * (Y) -/
     * @param context a GrCUDA context with the specified options
     */
    static void joinWithExtraKernel(Context context) {
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
        Value squareKernel = buildkernel.execute(SQUARE_WITH_CONST, "square", "const pointer, pointer, sint32");
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
