package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.LittleEndianNativeArrayView;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.profiles.ValueProfile;

public class DeviceArrayMock extends DeviceArray {
    public DeviceArrayMock() {
        this(0, Type.SINT32);
    }

    public DeviceArrayMock(int numElements) {
        this(numElements, Type.SINT32);
    }

    public DeviceArrayMock(int numElements, Type type) {
        super(new AsyncGrCUDAExecutionContextMock(), numElements, type);
    }

    public DeviceArrayMock(AbstractGrCUDAExecutionContext context) {
        super(context, 0, Type.SINT32);
        if (context.isArchitecturePascalOrNewer()) {
            this.addArrayUpToDateLocations(CPUDevice.CPU_DEVICE_ID);
        } else {
            this.addArrayUpToDateLocations(context.getCurrentGPU());
        }
    }

    @Override
    public void writeArrayElement(long index, Object value,
                                  @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
                                  @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws UnsupportedTypeException, InvalidArrayIndexException {
        if (this.canSkipSchedulingWrite()) {
            // Fast path, don't do anything here;
        } else {
            new DeviceArrayWriteExecutionMock(this, index, value).schedule();
        }
    }

    @Override
    protected LittleEndianNativeArrayView allocateMemory() {
        return null;
    }

    @Override
    public String toString() {
        return this.getElementType() + "[" + this.getArraySize() + "]";
    }
}

