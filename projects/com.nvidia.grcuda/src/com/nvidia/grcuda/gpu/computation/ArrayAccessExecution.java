package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.gpu.GrCUDAExecutionContext;

/**
 * Abstract class that wraps all computational elements representing accesses on managed memory by the CPU;
 */
public abstract class ArrayAccessExecution extends GrCUDAComputationalElement {
    public ArrayAccessExecution(GrCUDAExecutionContext grCUDAExecutionContext, InitializeArgumentSet initializer) {
        super(grCUDAExecutionContext, initializer);
    }

    @Override
    public boolean isComputationArrayAccess() { return true; }
}
