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
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.computation.prefetch.PrefetcherEnum;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

/**
 * Execute all computations synchronously, without computing dependencies or using streams;
 */
public class SyncGrCUDAExecutionContext extends AbstractGrCUDAExecutionContext {

    public SyncGrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, DependencyPolicyEnum dependencyPolicy, PrefetcherEnum inputPrefetch) {
        super(context, env, dependencyPolicy, inputPrefetch, ExecutionPolicyEnum.SYNC);
    }

    public SyncGrCUDAExecutionContext(CUDARuntime cudaRuntime, DependencyPolicyEnum dependencyPolicy) {
        super(cudaRuntime, dependencyPolicy, PrefetcherEnum.NONE, ExecutionPolicyEnum.SYNC);
    }

    public SyncGrCUDAExecutionContext(CUDARuntime cudaRuntime, DependencyPolicyEnum dependencyPolicy, PrefetcherEnum inputPrefetch) {
        super(cudaRuntime, dependencyPolicy, inputPrefetch, ExecutionPolicyEnum.SYNC);
    }

    /**
     * Register this computation for future execution by the {@link SyncGrCUDAExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data dependencies;
     */
    @Override
    public Object registerExecution(GrCUDAComputationalElement computation) throws UnsupportedTypeException {

        // Prefetching;
        arrayPrefetcher.prefetchToGpu(computation);

        // Book-keeping;
        computation.setComputationStarted();
        computation.updateIsComputationArrayAccess();

        // Start the computation immediately;
        Object result = computation.execute();

        // Wait for the computation to end;
        cudaRuntime.cudaDeviceSynchronize();

        return result;
    }

    /**
     * All computations are synchronous, and atomic;
     * @return false
     */
    @Override
    public boolean isAnyComputationActive() {
        return false;
    }
}
