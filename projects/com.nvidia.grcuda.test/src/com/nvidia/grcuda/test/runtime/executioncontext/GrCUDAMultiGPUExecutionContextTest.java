package com.nvidia.grcuda.test.runtime.executioncontext;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.test.util.GrCUDATestOptionsStruct;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assume.assumeTrue;

@RunWith(Parameterized.class)
public class GrCUDAMultiGPUExecutionContextTest {

    // FIXME: add multi-gpu policies;

    /**
     * Tests are executed for each of the {@link AsyncGrCUDAExecutionContext} values;
     * @return the current stream policy
     */

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return GrCUDATestUtil.getAllOptionCombinationsMultiGPU();
    }

    private final GrCUDATestOptionsStruct options;

    /**
     * Set to false if we discover that only a single GPU is available. Doing other tests is not useful;
     */
    private static boolean multipleGPUs = true;

    public GrCUDAMultiGPUExecutionContextTest(GrCUDATestOptionsStruct options) {
        this.options = options;
    }

    private static final int NUM_THREADS_PER_BLOCK = 32;

    private static final String SQUARE_KERNEL =
            "extern \"C\" __global__ void square(float* x, int n) {\n" +
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (idx < n) {\n" +
                    "       x[idx] = x[idx] * x[idx];\n" +
                    "    }" +
                    "}\n";

    @Before
    public void skipIfSingleGPU() {
        assumeTrue(multipleGPUs);
    }

    private boolean checkIfEnoughGPUsAreAvailable(Context context) {
        Value deviceCount = context.eval("grcuda", "cudaGetDeviceCount()");
        if (deviceCount.asInt() < 2) {
            // The system does not have multiple GPUs, skip all further multi-GPU tests;
            multipleGPUs = false;
            System.out.println("warning: only 1 GPU available, skipping further multi-GPU tests");
            return false;
        } else if (this.options.numberOfGPUs > deviceCount.asInt()) {
            // If the test asks for more GPUs than available, skip it;
            return false;
        }
        // We have enough GPUs for this test;
        return true;
    }

    ////////////////////////////////////////////////////////
    // Basic multi-GPU testing, with manual GPU selection //
    ////////////////////////////////////////////////////////

    /**
     * Execute 2 independent kernels, 2 times in a row, manually specifying the GPU for them;
     */
    @Test
    public void dependency2KernelsManualGPUChoiceTest() {
        int numOfGPUs = 2;
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options, numOfGPUs)) {

            assumeTrue(checkIfEnoughGPUsAreAvailable(context));

            Value setDevice = context.eval("grcuda", "cudaSetDevice");

            final int numElements = 10000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value[] inputs = {x, y};

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            assertNotNull(squareKernel);

            // init arrays with values
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
                y.setArrayElement(i, 4.0);
            }
            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            for (int i = 0; i < numOfGPUs; i++) {
                setDevice.execute(i);
                // Perform the computation, twice;
                configuredSquareKernel.execute(inputs[i], numElements);
                configuredSquareKernel.execute(inputs[i], numElements);
            }

            // Verify the output;
            assertEquals(16.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(256.0, y.getArrayElement(0).asFloat(), 0.1);
        }
    }

    ///////////////////////////////////////////////////////////
    // Basic multi-GPU testing, with automatic GPU selection //
    ///////////////////////////////////////////////////////////

    /**
     * Execute 2 independent kernels, 2 times in a row;
     */
    @Test
    public void dependency2KernelsSimpleTest() {
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {

            assumeTrue(checkIfEnoughGPUsAreAvailable(context));

            final int numElements = 10;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            assertNotNull(squareKernel);

            // init arrays with values
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
                y.setArrayElement(i, 4.0);
            }

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, numElements);
            configuredSquareKernel.execute(y, numElements);

            // Perform the computation;
            configuredSquareKernel.execute(x, numElements);
            configuredSquareKernel.execute(y, numElements);

            // Verify the output;
            assertEquals(16.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(256.0, y.getArrayElement(0).asFloat(), 0.1);
        }
    }

    /**
     * Test with 3 kernels: kernel0 does not have dependencies.
     * kernel1 is the parent of kernel2;
     */
    @Test
    public void dependencyKernelsTestA() {

        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {

            assumeTrue(checkIfEnoughGPUsAreAvailable(context));

            final int numElements = 10000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            assertNotNull(squareKernel);

            // init arrays with values
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
                y.setArrayElement(i, 4.0);
            }

            Value configuredK0 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredK1 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredK2 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredK0.execute(x, numElements);
            configuredK1.execute(y, numElements);

            // Perform the computation;
            configuredK2.execute(y, numElements);
            // Verify the output;
            assertEquals(4.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(256.0, y.getArrayElement(0).asFloat(), 0.1);
        }
    }

    //////////////////////////////////////////
    // Call existing tests, using multi-GPU //
    //////////////////////////////////////////

    /**
     * Test a join pattern (x) & (y) -> (z), with data in x and y being copied from other arrays;
     */
    @Test
    public void dependencyPipelineWithArrayCopyTest() {
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            assumeTrue(checkIfEnoughGPUsAreAvailable(context));
            GrCUDAComputationsWithGPU.arrayCopyWithJoin(context);
        }
    }

    @Test
    public void parallelKernelsWithReadOnlyArgsTest() {
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            assumeTrue(checkIfEnoughGPUsAreAvailable(context));
            GrCUDAComputationsWithGPU.parallelKernelsWithReadOnlyArgs(context);
        }
    }

    @Test
    public void simpleForkReadInputTest() {
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            assumeTrue(checkIfEnoughGPUsAreAvailable(context));
            GrCUDAComputationsWithGPU.simpleForkReadInput(context);
        }
    }

    @Test
    public void forkWithReadOnlyTest() {
        // Test a computation of form A(1) --> B(1r, 2)
        //                                 \-> C(1r, 3)
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            assumeTrue(checkIfEnoughGPUsAreAvailable(context));
            GrCUDAComputationsWithGPU.forkWithReadOnly(context);
        }
    }

    @Test
    public void dependencyPipelineDiamondTest() {
        // Test a computation of form A(1) --> B(1r, 2) -> D(1)
        //                                 \-> C(1r, 3) -/
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            assumeTrue(checkIfEnoughGPUsAreAvailable(context));
            GrCUDAComputationsWithGPU.dependencyPipelineDiamond(context);
        }
    }

    @Test
    public void joinWithExtraKernelTest() {
        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options)) {
            assumeTrue(checkIfEnoughGPUsAreAvailable(context));
            GrCUDAComputationsWithGPU.joinWithExtraKernel(context);
        }
    }
}