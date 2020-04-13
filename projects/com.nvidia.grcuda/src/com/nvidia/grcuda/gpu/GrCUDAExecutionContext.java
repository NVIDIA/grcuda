package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.array.AbstractArray;

import java.util.HashSet;
import java.util.Set;

/**
 * Class used to monitor the state of GrCUDA execution, keep track of memory allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public class GrCUDAExecutionContext {

    /**
     * Set that contains all the arrays allocated so far.
     */
    final private Set<AbstractArray> arraySet = new HashSet<>();

    public GrCUDAExecutionContext() {

    }

    public void registerArray(AbstractArray array) {
        arraySet.add(array);
        System.out.println("-- added array to context: " + System.identityHashCode(array) + "; " + array.toString());
    }
}
