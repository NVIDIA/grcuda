package com.nvidia.grcuda.runtime.computation.arraycomputation;

import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.computation.InitializeDependencyList;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

import java.util.Optional;

/**
 * Abstract class that wraps all computational elements representing accesses on managed memory by the CPU;
 */
public abstract class ArrayAccessExecution<T extends AbstractArray> extends GrCUDAComputationalElement {

    protected boolean isComputationArrayAccess = true;
    protected T array;

    public ArrayAccessExecution(AbstractGrCUDAExecutionContext grCUDAExecutionContext, InitializeDependencyList initializer, T array) {
        super(grCUDAExecutionContext, initializer);
        this.array = array;
    }

    @Override
    public void updateIsComputationArrayAccess() {
        this.array.setLastComputationArrayAccess(isComputationArrayAccess);
    }

    @Override
    protected Optional<CUDAStream> additionalStreamDependencyImpl() { return Optional.of(array.getStreamMapping()); }
}
