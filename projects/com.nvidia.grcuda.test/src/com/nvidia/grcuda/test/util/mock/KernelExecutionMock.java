package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Mock class to test the DAG execution;
 */
public class KernelExecutionMock extends GrCUDAComputationalElement {

    /**
     * Simulate an execution by forcing a wait that last the given number of milliseconds;
     */
    private int durationMs = 0;

    public KernelExecutionMock(AbstractGrCUDAExecutionContext grCUDAExecutionContext, List<ComputationArgumentWithValue> args) {
        super(grCUDAExecutionContext, args);
    }

    public KernelExecutionMock(AbstractGrCUDAExecutionContext grCUDAExecutionContext, List<ComputationArgumentWithValue> args, int durationMs) {
        super(grCUDAExecutionContext, args);
        this.durationMs = durationMs;
    }

    @Override
    public Object execute() {
        if (this.durationMs > 0) {
            try {
                Thread.sleep(this.durationMs);
            } catch (InterruptedException e) {
                System.out.println("ERROR; failed to pause " + this + " for " + this.durationMs + " msec");
                e.printStackTrace();
            }
        }
        return NoneValue.get();
    }

    @Override
    public boolean canUseStream() { return true; }

    @Override
    public void associateArraysToStreamImpl() { }

    @Override
    public String toString() {
        return "kernel mock" + "; args=[" +
                this.argumentList.stream().map(Object::toString).collect(Collectors.joining(", ")) +
                "]" + "; stream=" + this.getStream().getStreamNumber();
    }
}

