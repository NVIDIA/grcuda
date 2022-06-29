package com.nvidia.grcuda.test.runtime.array;

import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.MultiDimDeviceArray;
import com.nvidia.grcuda.runtime.array.MultiDimDeviceArrayView;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.test.util.mock.ArgumentMock;
import com.nvidia.grcuda.test.util.mock.AsyncGrCUDAExecutionContextMock;
import com.nvidia.grcuda.test.util.mock.DeviceArrayMock;
import com.nvidia.grcuda.test.util.mock.DeviceArrayReadExecutionMock;
import com.nvidia.grcuda.test.util.mock.DeviceArrayWriteExecutionMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.util.mock.KernelExecutionMock;
import com.nvidia.grcuda.test.util.mock.MultiDimDeviceArrayMock;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DeviceArrayLocationMockTest {

    @Test
    public void testIfInitializedCorrectlyPrePascal() {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder().setArchitecturePascalOrNewer(false).build();
        DeviceArray array1 = new DeviceArrayMock(context);
        DeviceArray array2 = new DeviceArrayMock(context);
        assertEquals(1, array1.getArrayUpToDateLocations().size());
        assertEquals(1, array2.getArrayUpToDateLocations().size());
        assertEquals(array1.getArrayUpToDateLocations(), array2.getArrayUpToDateLocations());
        assertTrue(array1.getArrayUpToDateLocations().contains(context.getCurrentGPU()));
    }

    @Test
    public void testIfInitializedCorrectlyPostPascal() {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder().setArchitecturePascalOrNewer(true).build();
        DeviceArray array1 = new DeviceArrayMock(context);
        DeviceArray array2 = new DeviceArrayMock(context);
        assertEquals(1, array1.getArrayUpToDateLocations().size());
        assertEquals(1, array2.getArrayUpToDateLocations().size());
        assertEquals(array1.getArrayUpToDateLocations(), array2.getArrayUpToDateLocations());
        assertTrue(array1.isArrayUpdatedInLocation(CPUDevice.CPU_DEVICE_ID));
    }

    @Test
    public void testIfLocationAdded() {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder()
                .setArchitecturePascalOrNewer(true)
                .setNumberOfAvailableGPUs(1)
                .setNumberOfGPUsToUse(1).build();
        DeviceArray array1 = new DeviceArrayMock(context);
        array1.addArrayUpToDateLocations(2);
        assertEquals(2, array1.getArrayUpToDateLocations().size());
        assertTrue(array1.isArrayUpdatedInLocation(2));
    }

    @Test
    public void testIfLocationReset() {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder().setArchitecturePascalOrNewer(true)
                .setNumberOfAvailableGPUs(1)
                .setNumberOfGPUsToUse(1).build();
        DeviceArray array1 = new DeviceArrayMock(context);
        array1.resetArrayUpToDateLocations(2);
        assertEquals(1, array1.getArrayUpToDateLocations().size());
        assertTrue(array1.isArrayUpdatedInLocation(2));
    }

    /**
     * Test that, when using multi-dimensional arrays, the array views' locations are propagated correctly;
     */
    @Test
    public void testMultiDimLocation() {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder().setArchitecturePascalOrNewer(true)
                .setNumberOfAvailableGPUs(2)
                .setNumberOfGPUsToUse(2).build();

        long[] dimensions = {2, 2};
        MultiDimDeviceArray array1 = new MultiDimDeviceArrayMock(context, dimensions, false);
        assertTrue(array1.isArrayUpdatedOnCPU());
        array1.addArrayUpToDateLocations(2);
        array1.addArrayUpToDateLocations(3);
        assertEquals(3, array1.getArrayUpToDateLocations().size());
        // Create a view, parameters don't matter;
        MultiDimDeviceArrayView view = new MultiDimDeviceArrayView(array1, 1, 0, 0);
        assertEquals(array1.getArrayUpToDateLocations(), view.getArrayUpToDateLocations());
        // Add a location to the view, check that it is propagated;
        view.addArrayUpToDateLocations(4);
        assertEquals(4, view.getArrayUpToDateLocations().size());
        assertEquals(array1.getArrayUpToDateLocations(), view.getArrayUpToDateLocations());
        // Reset locations on the view, check that the parent is updated;
        view.resetArrayUpToDateLocations(10);
        assertEquals(1, view.getArrayUpToDateLocations().size());
        assertEquals(array1.getArrayUpToDateLocations(), view.getArrayUpToDateLocations());
        // Reset locations on the parent, check that the view is updated;
        array1.resetArrayUpToDateLocations(CPUDevice.CPU_DEVICE_ID);
        assertEquals(1, view.getArrayUpToDateLocations().size());
        assertTrue(view.isArrayUpdatedOnCPU());
        assertTrue(view.isArrayUpdatedInLocation(CPUDevice.CPU_DEVICE_ID));
        assertEquals(array1.getArrayUpToDateLocations(), view.getArrayUpToDateLocations());
        // Reset locations on the view (again), but also on the parent, and check consistency;
        view.resetArrayUpToDateLocations(10);
        array1.addArrayUpToDateLocations(2);
        assertEquals(2, view.getArrayUpToDateLocations().size());
        assertEquals(array1.getArrayUpToDateLocations(), view.getArrayUpToDateLocations());
    }

    /**
     * Test that the location of arrays in a complex DAG is propagated correctly, also when using 2 GPUs;
     */
    @Test
    public void complexFrontierWithSyncMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setArchitecturePascalOrNewer(true)
                .setNumberOfAvailableGPUs(2)
                .setNumberOfGPUsToUse(2).build();

        DeviceArray array1 = new DeviceArrayMock(context);
        DeviceArray array2 = new DeviceArrayMock(context);
        DeviceArray array3 = new DeviceArrayMock(context);
        // K1(const A1, A2);
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(array1, true), new ArgumentMock(array2, false))).schedule();
        assertEquals(2, array1.getArrayUpToDateLocations().size());
        assertTrue(array1.isArrayUpdatedInLocation(0));
        assertTrue(array1.isArrayUpdatedOnCPU());
        assertEquals(1, array2.getArrayUpToDateLocations().size());
        assertTrue(array2.isArrayUpdatedInLocation(0));
        // Set another GPU;
        // K2(const A1, A3);
        context.setCurrentGPU(1);
        KernelExecutionMock k2 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(array1, true), new ArgumentMock(array3, false)));
        k2.setStream(new CUDAStream(0, 0, context.getCurrentGPU()));
        k2.schedule();
        assertEquals(3, array1.getArrayUpToDateLocations().size());
        assertTrue(array1.isArrayUpdatedInLocation(1));
        assertEquals(1, array3.getArrayUpToDateLocations().size());
        assertTrue(array3.isArrayUpdatedInLocation(1));
        assertEquals(1, array2.getArrayUpToDateLocations().size());  // A2 is unmodified;
        assertTrue(array2.isArrayUpdatedInLocation(0));
        // Write on 2 arrays, read on another array. The CPU will be the exclusive owner of the first 2 arrays,
        // and share with GPU 0 the other array;
        new DeviceArrayWriteExecutionMock(array1, 0, 0).schedule();
        new DeviceArrayReadExecutionMock(array2, 0).schedule();
        new DeviceArrayWriteExecutionMock(array3, 0, 0).schedule();
        assertEquals(1, array1.getArrayUpToDateLocations().size());
        assertTrue(array1.isArrayUpdatedOnCPU());
        assertEquals(1, array3.getArrayUpToDateLocations().size());
        assertTrue(array3.isArrayUpdatedOnCPU());
        assertEquals(2, array2.getArrayUpToDateLocations().size());
        assertTrue(array2.isArrayUpdatedOnCPU());
        assertTrue(array2.isArrayUpdatedInLocation(0));
    }
}
