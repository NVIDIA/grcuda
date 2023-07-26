package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.LittleEndianNativeArrayView;
import com.nvidia.grcuda.runtime.array.MultiDimDeviceArray;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;

public class MultiDimDeviceArrayMock extends MultiDimDeviceArray {
    public MultiDimDeviceArrayMock(long[] dimensions, boolean columnMajor) {
        super(new AsyncGrCUDAExecutionContextMock(), Type.SINT32, dimensions, columnMajor);
    }

    public MultiDimDeviceArrayMock(AbstractGrCUDAExecutionContext context, long[] dimensions, boolean columnMajor) {
        super(context,  Type.SINT32, dimensions, columnMajor);
        if (context.isArchitecturePascalOrNewer()) {
            this.addArrayUpToDateLocations(CPUDevice.CPU_DEVICE_ID);
        } else {
            this.addArrayUpToDateLocations(context.getCurrentGPU());
        }
    }

    @Override
    protected LittleEndianNativeArrayView allocateMemory() {
        return null;
    }
}
