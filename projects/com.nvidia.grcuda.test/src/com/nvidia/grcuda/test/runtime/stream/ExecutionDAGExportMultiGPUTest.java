package com.nvidia.grcuda.test.runtime.stream;

import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.executioncontext.GraphExport;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.executeMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.executeMockComputationAndValidate;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.forkJoinMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.hitsMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.imageMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.joinPipeline2MockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.joinPipeline3MockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.joinPipeline4MockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.joinPipelineMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.manyIndependentKernelsMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.manyKernelsMockComputation;
import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class ExecutionDAGExportMultiGPUTest{

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {RetrieveNewStreamPolicyEnum.ALWAYS_NEW, RetrieveNewStreamPolicyEnum.REUSE}
        }));
    }

    private final RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy;

    public ExecutionDAGExportMultiGPUTest (RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy) {
        this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
    }

    private final static int IMAGE_NUM_STREAMS = 4;
    private final static int HITS_NUM_STREAMS = 2;

    // Test the STREAM_AWARE policy on 2 and 3 GPUs, on the image pipeline and HITS DAGs.
    // In each case, validate the mapping of each computation on the right GPUs,
    // and the total number of streams created;

    @Test
    public void lessBusyWithThreeGPUImageTest() throws UnsupportedTypeException {
        int numGPU = 3;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(imageMockComputation(context),
                Arrays.asList(
                        0, 1, 2,
                        0, 1,
                        2, 0,
                        2, 2, 1, 0, 0));
        graphExport(context.getDag(), "lessBusyWithThreeGPUImageTest");
    }

    @Test
    public void lessBusyWithTwoGPUHitsTest() throws UnsupportedTypeException {
        int numGPU = 2;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(hitsMockComputation(context),
                Arrays.asList(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0));
        graphExport(context.getDag(), "lessBusyWithTwoGPUHitsTest");
    }

    @Test
    public void lessBusyManyKernelsWithFourGPUTest() throws UnsupportedTypeException {
        int numGPU = 4;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.STREAM_AWARE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(manyKernelsMockComputation(context),
                Arrays.asList(0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 0, 0, 2, 1));
        assertEquals(6, context.getStreamManager().getNumberOfStreams());
        graphExport(context.getDag(), "lessBusyManyKernelsWithFourGPUTest");
    }

    @Test
    public void roundRobinTest() throws UnsupportedTypeException {
        int[] gpus = {1, 4, 8};
        for (int numGPU : gpus) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.ROUND_ROBIN)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
            executeMockComputationAndValidate(manyIndependentKernelsMockComputation(context),
                    Arrays.asList(0, 1 % numGPU, 2 % numGPU, 3 % numGPU, 4 % numGPU, 5 % numGPU, 6 % numGPU, 7 % numGPU, 8 % numGPU, 9 % numGPU));
            graphExport(context.getDag(), "roundRobinTest" + numGPU + "GPU");
        }
    }

    @Test
    public void roundRobinForkJoinWithTwoGPUTest() throws UnsupportedTypeException {
        int numGPU = 2;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.ROUND_ROBIN)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(forkJoinMockComputation(context),
                Arrays.asList(0, 1, 0, 0, 0));
        graphExport(context.getDag(), "roundRobinForkJoinWithTwoGPUTest");
    }

    @Test
    public void minTransferWithDepTest() throws UnsupportedTypeException {
        int[] gpus = {4, 8};
        for (int numGPU : gpus) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
            executeMockComputationAndValidate(joinPipelineMockComputation(context),
                    Arrays.asList(0, 1, 2, 3, 0, 2, 3));
            graphExport(context.getDag(), "minTransferWithDepTest" + numGPU + "GPU");
        }
    }

    @Test
    public void minTransferWithThreeGPUImageTest() throws UnsupportedTypeException {
        int numGPU = 3;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        executeMockComputationAndValidate(imageMockComputation(context),
                Arrays.asList(
                        0, 0, 0,
                        0, 0,
                        0, 0,
                        0, 0, 0, 0, 0));
        graphExport(context.getDag(), "minTransferWithThreeGPUImageTest");
    }

    @Test
    public void minTransferManyKernelsWithFourGPUTest() throws UnsupportedTypeException {
        int numGPU = 4;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        // The last 4 computations are scheduled on GPU0 as all devices contain just 1 required array and GPU0 is first;
        executeMockComputationAndValidate(manyKernelsMockComputation(context),
                Arrays.asList(0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 0, 0, 0, 0));
        graphExport(context.getDag(), "minTransferManyKernelsWithFourGPUTest");
    }

    @Test
    public void minTransferDisjointWithDep4MultiGPUTest() throws UnsupportedTypeException {
        int[] gpus = {4, 8};
        for (int numGPU : gpus) {
            AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                    .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                    .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_DISJOINT)
                    .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                    .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                    .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
            // Computation 5/7 is scheduled on 1 because GPU0 is not considered a parent,
            //   A is read-only in both Comp1 and Comp5;
            executeMockComputationAndValidate(joinPipeline4MockComputation(context),
                    Arrays.asList(0, 1, 2, 3, 1, 3, 3));
            graphExport(context.getDag(), "minTransferDisjointWithDep4MultiGPUTest" + numGPU + "GPU");
        }
    }

    @Test
    public void minTransferDisjointManyKernelsWithFourGPUTest() throws UnsupportedTypeException {
        int numGPU = 4;
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.MULTIGPU_DISJOINT)
                .setDeviceSelectionPolicy(DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE)
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setNumberOfGPUsToUse(numGPU).setNumberOfAvailableGPUs(numGPU).build();
        // The last 4 computations are scheduled on GPU0 as all devices contain just 1 required array and GPU0 is first;
        executeMockComputationAndValidate(manyKernelsMockComputation(context),
                Arrays.asList(0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 0, 0, 2, 2));
        graphExport(context.getDag(), "minTransferDisjointManyKernelsWithFourGPUTest");
    }


    public void graphExport(ExecutionDAG dag, String name){
        GraphExport graphExport = new GraphExport(dag);

//        if (retrieveNewStreamPolicy==RetrieveNewStreamPolicyEnum.ALWAYS_NEW){
//            graphExport.graphGenerator("../" + name + "AlwaysNew");
//        } else {
//            graphExport.graphGenerator("../" + name + "Reuse");
//        }
    }
}
