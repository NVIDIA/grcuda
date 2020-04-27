package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.GrCUDAExecutionContext;

/**
 * Abstract class that wraps all computational elements representing accesses on managed memory by the CPU;
 */
public abstract class ArrayAccessExecution<T extends AbstractArray> extends GrCUDAComputationalElement {

    protected boolean isComputationArrayAccess = true;
    protected T array;

    public ArrayAccessExecution(GrCUDAExecutionContext grCUDAExecutionContext, InitializeArgumentSet initializer, T array) {
        super(grCUDAExecutionContext, initializer);
        this.array = array;
    }

    @Override
    public void updateIsComputationArrayAccess() {
        this.array.setLastComputationArrayAccess(isComputationArrayAccess);
    }
}
