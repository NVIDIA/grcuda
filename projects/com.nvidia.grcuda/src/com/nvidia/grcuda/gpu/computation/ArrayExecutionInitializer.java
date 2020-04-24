package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.array.AbstractArray;
import com.oracle.truffle.api.CompilerDirectives;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * The only argument in {@link com.nvidia.grcuda.array.AbstractArray} computations is the array itself.
 * Note that in {@link com.nvidia.grcuda.array.MultiDimDeviceArrayView} the array is the parent {@link com.nvidia.grcuda.array.MultiDimDeviceArray},
 * while in {@link com.nvidia.grcuda.array.MultiDimDeviceArray} there is currently no need to explicitly represent computations,
 * as they cannot directly the underlying memory;
 */
class ArrayExecutionInitializer implements InitializeArgumentSet {

    private final AbstractArray array;

    ArrayExecutionInitializer(AbstractArray array) {
        this.array = array;
    }

    @Override
    @CompilerDirectives.TruffleBoundary
    public Set<Object> initialize() {
        return new HashSet<>(Collections.singleton(this.array));
    }
}
