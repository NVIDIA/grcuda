package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.gpu.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Mock class that represents a synchronous execution,
 * it can be used to synchronize previous computations using the specified arguments;
 */
public class SyncExecutionMock extends GrCUDAComputationalElement {

    public SyncExecutionMock(GrCUDAExecutionContext grCUDAExecutionContext, List<ComputationArgumentWithValue> args) {
        super(grCUDAExecutionContext, args);
    }

    @Override
    public Object execute() {
        this.setComputationFinished();
        return NoneValue.get();
    }

    @Override
    public boolean canUseStream() { return false; }

    @Override
    public void associateArraysToStreamImpl() { }

    @Override
    public String toString() {
        return "sync" + "; args=[" +
                this.argumentList.stream().map(Object::toString).collect(Collectors.joining(", ")) +
                "]";
    }
}

