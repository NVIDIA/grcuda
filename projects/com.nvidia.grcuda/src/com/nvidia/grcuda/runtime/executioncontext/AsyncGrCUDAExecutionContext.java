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
package com.nvidia.grcuda.runtime.executioncontext;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDALogger;
import com.nvidia.grcuda.GrCUDAOptionMap;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.DeviceList;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.computation.prefetch.AsyncArrayPrefetcher;
import com.nvidia.grcuda.runtime.stream.GrCUDAStreamManager;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

/**
 * Class used to monitor the state of GrCUDA execution, keep track of memory allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public class AsyncGrCUDAExecutionContext extends AbstractGrCUDAExecutionContext {

    /**
     * Reference to the {@link com.nvidia.grcuda.runtime.stream.GrCUDAStreamManager} that takes care of
     * scheduling computations on different streams;
     */
    private final GrCUDAStreamManager streamManager;

    public AsyncGrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env) {
        this(new CUDARuntime(context, env), context.getOptions());
    }

    public AsyncGrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAOptionMap options) {
        this(cudaRuntime, options, new GrCUDAStreamManager(cudaRuntime, options));
    }

    public AsyncGrCUDAExecutionContext(CUDARuntime cudaRuntime, GrCUDAOptionMap options, GrCUDAStreamManager streamManager) {
        super(cudaRuntime, options);
        this.streamManager = streamManager;
        // Compute if we should use a prefetcher;
        if (options.isInputPrefetch() && this.cudaRuntime.isArchitectureIsPascalOrNewer()) {
            arrayPrefetcher = new AsyncArrayPrefetcher(this.cudaRuntime);
        }
    }

    /**
     * Register this computation for future execution by the {@link AsyncGrCUDAExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data dependencies;
     */
    @Override
    public Object registerExecution(GrCUDAComputationalElement computation) throws UnsupportedTypeException {
        // Add the new computation to the DAG
        ExecutionDAG.DAGVertex vertex = dag.append(computation);

        // Compute the stream where the computation will be done, if the computation can be performed asynchronously;
        streamManager.assignStream(vertex);

        // Prefetching;
        arrayPrefetcher.prefetchToGpu(vertex);

        // Start the computation;
        Object result = executeComputation(vertex);

        // Associate a CUDA event to this computation, if performed asynchronously;
        streamManager.assignEventStop(vertex);

        GrCUDALogger.getLogger(GrCUDALogger.EXECUTIONCONTEXT_LOGGER).finest(() -> "-- running " + vertex.getComputation());

        return result;
    }

    @Override
    public DeviceList getDeviceList() {
        // The device list is created only once, and we always return the same device list object.
        // This is just an optimization to avoid creating new objects;
        return this.getStreamManager().getDeviceList();
    }

    @Override
    public Device getDevice(int deviceId) {
        // The device list is created only once, and we always return the same device object.
        // This is just an optimization to avoid creating new objects;
        return this.getStreamManager().getDevice(deviceId);
    }

    @Override
    public boolean isAnyComputationActive() {
        return this.streamManager.isAnyComputationActive();
    }

    public GrCUDAStreamManager getStreamManager() {
        return streamManager;
    }

    /**
     * Delete internal structures that require manual cleanup operations;
     */
    @Override
    public void cleanup() {
        streamManager.cleanup();
    }

    private Object executeComputation(ExecutionDAG.DAGVertex vertex) throws UnsupportedTypeException {
        // Before starting this computation, ensure that all its parents have finished their computation;
        streamManager.syncParentStreams(vertex);

        // Perform the computation;
        vertex.getComputation().setComputationStarted();

        // For all input arrays, update whether this computation is an array access done by the CPU;
        vertex.getComputation().updateLocationOfArrays();

        // Associate a CUDA event to the starting phase of the computation in order to get the Elapsed time from start to the end
        streamManager.assignEventStart(vertex);

        return vertex.getComputation().execute();
    }
}
