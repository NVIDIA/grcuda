package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.array.DeviceArray;
import com.oracle.truffle.api.CompilerDirectives;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * The only argument in {@link DeviceArray} computations is the array itself;
 */
class DeviceArrayExecutionInitializer implements InitializeArgumentSet {

    private final DeviceArray array;

    DeviceArrayExecutionInitializer(DeviceArray array) {
        this.array = array;
    }

    @Override
    @CompilerDirectives.TruffleBoundary
    public Set<Object> initialize() {
        return new HashSet<>(Collections.singleton(this.array));
    }
}
