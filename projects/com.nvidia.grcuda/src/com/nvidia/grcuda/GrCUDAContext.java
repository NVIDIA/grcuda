/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
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
package com.nvidia.grcuda;

import com.nvidia.grcuda.cublas.CUBLASRegistry;
import com.nvidia.grcuda.cuml.CUMLRegistry;
import com.nvidia.grcuda.functions.BindFunction;
import com.nvidia.grcuda.functions.BindKernelFunction;
import com.nvidia.grcuda.functions.BuildKernelFunction;
import com.nvidia.grcuda.functions.DeviceArrayFunction;
import com.nvidia.grcuda.functions.GetDeviceFunction;
import com.nvidia.grcuda.functions.GetDevicesFunction;
import com.nvidia.grcuda.functions.map.MapFunction;
import com.nvidia.grcuda.functions.map.ShredFunction;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.computation.prefetch.PrefetcherEnum;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.executioncontext.MultithreadGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.executioncontext.SyncGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.TruffleLanguage.Env;
import org.graalvm.options.OptionKey;

import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Context for the grCUDA language holds reference to CUDA runtime, a function registry and device
 * resources.
 */
public final class GrCUDAContext {

    public static final ExecutionPolicyEnum DEFAULT_EXECUTION_POLICY = ExecutionPolicyEnum.DEFAULT;
    public static final DependencyPolicyEnum DEFAULT_DEPENDENCY_POLICY = DependencyPolicyEnum.DEFAULT;
    public static final RetrieveNewStreamPolicyEnum DEFAULT_RETRIEVE_STREAM_POLICY = RetrieveNewStreamPolicyEnum.FIFO;
    public static final RetrieveParentStreamPolicyEnum DEFAULT_PARENT_STREAM_POLICY = RetrieveParentStreamPolicyEnum.DEFAULT;
    public static final boolean DEFAULT_FORCE_STREAM_ATTACH = false;

    private static final String ROOT_NAMESPACE = "CU";

    private final Env env;
    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final Namespace rootNamespace;
    private final ArrayList<Runnable> disposables = new ArrayList<>();
    private final AtomicInteger moduleId = new AtomicInteger(0);
    private volatile boolean cudaInitialized = false;
    private final RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy;
    private final RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum;
    private final boolean forceStreamAttach;
    private final boolean inputPrefetch;

    // this is used to look up pre-existing call targets for "map" operations, see MapArrayNode
    private final ConcurrentHashMap<Class<?>, CallTarget> uncachedMapCallTargets = new ConcurrentHashMap<>();

    public GrCUDAContext(Env env) {
        this.env = env;

        // Retrieve if we should force array stream attachment;
        forceStreamAttach = env.getOptions().get(GrCUDAOptions.ForceStreamAttach);

        // Retrieve if we should prefetch input data to GPU;
        inputPrefetch = env.getOptions().get(GrCUDAOptions.InputPrefetch);

        // Retrieve the stream retrieval policy;
        retrieveNewStreamPolicy = parseRetrieveStreamPolicy(env.getOptions().get(GrCUDAOptions.RetrieveNewStreamPolicy));
        
        // Retrieve how streams are obtained from parent computations;
        retrieveParentStreamPolicyEnum = parseParentStreamPolicy(env.getOptions().get(GrCUDAOptions.RetrieveParentStreamPolicy));

        // Retrieve the dependency computation policy;
        DependencyPolicyEnum dependencyPolicy = parseDependencyPolicy(env.getOptions().get(GrCUDAOptions.DependencyPolicy));
        System.out.println("-- using " + dependencyPolicy.getName() + " dependency policy");

        // Retrieve the execution policy;
        ExecutionPolicyEnum executionPolicy = parseExecutionPolicy(env.getOptions().get(GrCUDAOptions.ExecutionPolicy));
        // Initialize the execution policy;
        System.out.println("-- using " + executionPolicy.getName() + " execution policy");
        switch (executionPolicy) {
            case SYNC:
                this.grCUDAExecutionContext = new SyncGrCUDAExecutionContext(this, env, dependencyPolicy, PrefetcherEnum.SYNC);
                break;
            case DEFAULT:
                this.grCUDAExecutionContext = new GrCUDAExecutionContext(this, env ,dependencyPolicy, PrefetcherEnum.DEFAULT);
                break;
            default:
                this.grCUDAExecutionContext = new GrCUDAExecutionContext(this, env, dependencyPolicy, PrefetcherEnum.DEFAULT);
        }

        Namespace namespace = new Namespace(ROOT_NAMESPACE);
        namespace.addNamespace(namespace);
        namespace.addFunction(new BindFunction());
        namespace.addFunction(new DeviceArrayFunction(this.grCUDAExecutionContext));
        namespace.addFunction(new MapFunction());
        namespace.addFunction(new ShredFunction());
        namespace.addFunction(new BindKernelFunction(this.grCUDAExecutionContext));
        namespace.addFunction(new BuildKernelFunction(this.grCUDAExecutionContext));
        namespace.addFunction(new GetDevicesFunction(this.grCUDAExecutionContext.getCudaRuntime()));
        namespace.addFunction(new GetDeviceFunction(this.grCUDAExecutionContext.getCudaRuntime()));
        this.grCUDAExecutionContext.getCudaRuntime().registerCUDAFunctions(namespace);
        if (this.getOption(GrCUDAOptions.CuMLEnabled)) {
            Namespace ml = new Namespace(CUMLRegistry.NAMESPACE);
            namespace.addNamespace(ml);
            new CUMLRegistry(this).registerCUMLFunctions(ml);
        }
        if (this.getOption(GrCUDAOptions.CuBLASEnabled)) {
            Namespace blas = new Namespace(CUBLASRegistry.NAMESPACE);
            namespace.addNamespace(blas);
            new CUBLASRegistry(this).registerCUBLASFunctions(blas);
        }
        this.rootNamespace = namespace;
    }

    public Env getEnv() {
        return env;
    }

    public AbstractGrCUDAExecutionContext getGrCUDAExecutionContext() {
        return grCUDAExecutionContext;
    }

    public CUDARuntime getCUDARuntime() {
        return this.grCUDAExecutionContext.getCudaRuntime();
    }

    public Namespace getRootNamespace() {
        return rootNamespace;
    }

    public void addDisposable(Runnable disposable) {
        disposables.add(disposable);
    }

    public void disposeAll() {
        for (Runnable runnable : disposables) {
            runnable.run();
        }
    }

    public int getNextModuleId() {
        return moduleId.incrementAndGet();
    }

    public boolean isCUDAInitialized() {
        return cudaInitialized;
    }

    public void setCUDAInitialized() {
        cudaInitialized = true;
    }

    public ConcurrentHashMap<Class<?>, CallTarget> getMapCallTargets() {
        return uncachedMapCallTargets;
    }

    public RetrieveNewStreamPolicyEnum getRetrieveNewStreamPolicy() {
        return retrieveNewStreamPolicy;
    }
    
    public RetrieveParentStreamPolicyEnum getRetrieveParentStreamPolicyEnum() {
        return retrieveParentStreamPolicyEnum;
    }

    public boolean isForceStreamAttach() {
        return forceStreamAttach;
    }

    /**
     * Compute the maximum number of concurrent threads that can be spawned by GrCUDA.
     * This value is usually smaller or equal than the number of logical CPU threads available on the machine.
     * @return the maximum number of concurrent threads that can be spawned by GrCUDA
     */
    public int getNumberOfThreads() {
        return Runtime.getRuntime().availableProcessors();
    }

    @TruffleBoundary
    public <T> T getOption(OptionKey<T> key) {
        return env.getOptions().get(key);
    }

    @TruffleBoundary
    private static ExecutionPolicyEnum parseExecutionPolicy(String policyString) {
        switch(policyString) {
            case "sync":
                return ExecutionPolicyEnum.SYNC;
            case "default":
                return ExecutionPolicyEnum.DEFAULT;
            default:
                System.out.println("Warning: unknown execution policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_EXECUTION_POLICY);
                return GrCUDAContext.DEFAULT_EXECUTION_POLICY;
        }
    }

    @TruffleBoundary
    private static DependencyPolicyEnum parseDependencyPolicy(String policyString) {
        switch(policyString) {
            case "with-const":
                return DependencyPolicyEnum.WITH_CONST;
            case "default":
                return DependencyPolicyEnum.DEFAULT;
            default:
                System.out.println("Warning: unknown dependency policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_DEPENDENCY_POLICY);
                return GrCUDAContext.DEFAULT_DEPENDENCY_POLICY;
        }
    }

    @TruffleBoundary
    private static RetrieveNewStreamPolicyEnum parseRetrieveStreamPolicy(String policyString) {
        switch(policyString) {
            case "fifo":
                return RetrieveNewStreamPolicyEnum.FIFO;
            case "always-new":
                return RetrieveNewStreamPolicyEnum.ALWAYS_NEW;
            default:
                System.out.println("Warning: unknown new stream retrieval policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_RETRIEVE_STREAM_POLICY);
                return GrCUDAContext.DEFAULT_RETRIEVE_STREAM_POLICY;
        }
    }
    @TruffleBoundary
    private static RetrieveParentStreamPolicyEnum parseParentStreamPolicy(String policyString) {
        switch(policyString) {
            case "disjoint":
                return RetrieveParentStreamPolicyEnum.DISJOINT;
            case "default":
                return RetrieveParentStreamPolicyEnum.DEFAULT;
            default:
                System.out.println("Warning: unknown parent stream retrieval policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_PARENT_STREAM_POLICY);
                return GrCUDAContext.DEFAULT_PARENT_STREAM_POLICY;
        }
    }


    /**
     * Cleanup the GrCUDA context at the end of the execution;
     */
    public void cleanup() {
        this.grCUDAExecutionContext.cleanup();
    }
}
