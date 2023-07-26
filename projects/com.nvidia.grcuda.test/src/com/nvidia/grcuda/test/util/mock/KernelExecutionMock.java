/*
 * Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
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
    private final int durationMs;

    private final String name;

    public KernelExecutionMock(AbstractGrCUDAExecutionContext grCUDAExecutionContext, List<ComputationArgumentWithValue> args) {
        this(grCUDAExecutionContext, args, "kernel");
    }

    public KernelExecutionMock(AbstractGrCUDAExecutionContext grCUDAExecutionContext, List<ComputationArgumentWithValue> args, String name) {
        this(grCUDAExecutionContext, args, name, 0);
    }

    public KernelExecutionMock(AbstractGrCUDAExecutionContext grCUDAExecutionContext, List<ComputationArgumentWithValue> args, String name, int durationMs) {
        super(grCUDAExecutionContext, args);
        this.name = name;
        this.durationMs = durationMs;
    }

    public String getName() {
        return name;
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
        return this.getName() + ": args={" +
                this.argumentsThatCanCreateDependencies.stream().map(Object::toString).collect(Collectors.joining(", ")) +
                "}" + "; stream=" + this.getStream().getStreamNumber() + "; gpu=" + this.getStream().getStreamDeviceId();
    }
}

