/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
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
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
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
package com.nvidia.grcuda;

import com.nvidia.grcuda.cudalibraries.cublas.CUBLASRegistry;
import com.nvidia.grcuda.cudalibraries.cuml.CUMLRegistry;
import com.nvidia.grcuda.functions.BindAllFunction;
import com.nvidia.grcuda.functions.BindFunction;
import com.nvidia.grcuda.functions.BindKernelFunction;
import com.nvidia.grcuda.functions.BuildKernelFunction;
import com.nvidia.grcuda.functions.DeviceArrayFunction;
import com.nvidia.grcuda.functions.GetDeviceFunction;
import com.nvidia.grcuda.functions.GetDevicesFunction;
import com.nvidia.grcuda.functions.map.MapFunction;
import com.nvidia.grcuda.functions.map.ShredFunction;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.computation.prefetch.PrefetcherEnum;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.executioncontext.SyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.cudalibraries.tensorrt.TensorRTRegistry;
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.TruffleLanguage.Env;
import com.oracle.truffle.api.TruffleLogger;
import org.graalvm.options.OptionKey;

import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Context for the GrCUDA language holds reference to CUDA runtime, a function registry and device
 * resources.
 */
public final class GrCUDAContext {

    public static final ExecutionPolicyEnum DEFAULT_EXECUTION_POLICY = ExecutionPolicyEnum.ASYNC;
    public static final DependencyPolicyEnum DEFAULT_DEPENDENCY_POLICY = DependencyPolicyEnum.NO_CONST;
    public static final RetrieveNewStreamPolicyEnum DEFAULT_RETRIEVE_STREAM_POLICY = RetrieveNewStreamPolicyEnum.FIFO;
    public static final RetrieveParentStreamPolicyEnum DEFAULT_PARENT_STREAM_POLICY = RetrieveParentStreamPolicyEnum.SAME_AS_PARENT;
    public static final boolean DEFAULT_FORCE_STREAM_ATTACH = false;

    private static final String ROOT_NAMESPACE = "CU";

    private static final TruffleLogger LOGGER = TruffleLogger.getLogger(GrCUDALanguage.ID, "com.nvidia.grcuda.GrCUDAContext");

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
    private final boolean enableMultiGPU;

    // this is used to look up pre-existing call targets for "map" operations, see MapArrayNode
    private final ConcurrentHashMap<Class<?>, CallTarget> uncachedMapCallTargets = new ConcurrentHashMap<>();

    public GrCUDAContext(Env env) {
        this.env = env;
        
        // Retrieve if we should force array stream attachment;
        forceStreamAttach = env.getOptions().get(GrCUDAOptions.ForceStreamAttach);

        // Retrieve if we should prefetch input data to GPU;
        inputPrefetch = env.getOptions().get(GrCUDAOptions.InputPrefetch);

        // See if we allow the use of multiple GPUs in the system;
        enableMultiGPU = env.getOptions().get(GrCUDAOptions.EnableMultiGPU);

        // Retrieve the stream retrieval policy;
        retrieveNewStreamPolicy = parseRetrieveStreamPolicy(env.getOptions().get(GrCUDAOptions.RetrieveNewStreamPolicy));
        
        // Retrieve how streams are obtained from parent computations;
        retrieveParentStreamPolicyEnum = parseParentStreamPolicy(env.getOptions().get(GrCUDAOptions.RetrieveParentStreamPolicy));

        // Retrieve the dependency computation policy;
        DependencyPolicyEnum dependencyPolicy = parseDependencyPolicy(env.getOptions().get(GrCUDAOptions.DependencyPolicy));
        LOGGER.fine("using " + dependencyPolicy.getName() + " dependency policy");

        // Retrieve the execution policy;
        ExecutionPolicyEnum executionPolicy = parseExecutionPolicy(env.getOptions().get(GrCUDAOptions.ExecutionPolicy));
        
        
        // FIXME: TensorRT is currently incompatible with the async scheduler. TensorRT is supported in CUDA 11.4, and we cannot test it. 
        //  Once Nvidia adds support for it, we want to remove this limitation;
        if (this.getOption(GrCUDAOptions.TensorRTEnabled) && executionPolicy == ExecutionPolicyEnum.ASYNC) {
            LOGGER.warning("TensorRT and the asynchronous scheduler are not compatible. Switching to the synchronous scheduler.");
            executionPolicy = ExecutionPolicyEnum.SYNC;
        }
        
        // Initialize the execution policy;
        LOGGER.fine("using" + executionPolicy.getName() + " execution policy");
        switch (executionPolicy) {
            case SYNC:
                this.grCUDAExecutionContext = new SyncGrCUDAExecutionContext(this, env, dependencyPolicy, inputPrefetch ? PrefetcherEnum.SYNC : PrefetcherEnum.NONE);
                break;
            case ASYNC:
                this.grCUDAExecutionContext = new GrCUDAExecutionContext(this, env ,dependencyPolicy, inputPrefetch ? PrefetcherEnum.ASYNC : PrefetcherEnum.NONE);
                break;
            default:
                this.grCUDAExecutionContext = new GrCUDAExecutionContext(this, env, dependencyPolicy, inputPrefetch ? PrefetcherEnum.ASYNC : PrefetcherEnum.NONE);
        }

        Namespace namespace = new Namespace(ROOT_NAMESPACE);
        namespace.addNamespace(namespace);
        namespace.addFunction(new BindFunction());
        namespace.addFunction(new DeviceArrayFunction(this.grCUDAExecutionContext));
        namespace.addFunction(new BindAllFunction(this));
        namespace.addFunction(new MapFunction());
        namespace.addFunction(new ShredFunction());
        namespace.addFunction(new BindKernelFunction(this.grCUDAExecutionContext));
        namespace.addFunction(new BuildKernelFunction(this.grCUDAExecutionContext));
        namespace.addFunction(new GetDevicesFunction(this.grCUDAExecutionContext.getCudaRuntime()));
        namespace.addFunction(new GetDeviceFunction(this.grCUDAExecutionContext.getCudaRuntime()));
        this.grCUDAExecutionContext.getCudaRuntime().registerCUDAFunctions(namespace);
        if (this.getOption(GrCUDAOptions.CuMLEnabled)) {
            if (this.getCUDARuntime().isArchitectureIsPascalOrNewer()) {
                Namespace ml = new Namespace(CUMLRegistry.NAMESPACE);
                namespace.addNamespace(ml);
                new CUMLRegistry(this).registerCUMLFunctions(ml);
            } else {
                LOGGER.warning("cuML is supported only on GPUs with compute capability >= 6.0 (Pascal and newer). It cannot be enabled.");
            }
        }
        if (this.getOption(GrCUDAOptions.CuBLASEnabled)) {
                Namespace blas = new Namespace(CUBLASRegistry.NAMESPACE);
                namespace.addNamespace(blas);
                new CUBLASRegistry(this).registerCUBLASFunctions(blas);
        }
        if (this.getOption(GrCUDAOptions.TensorRTEnabled)) {
            Namespace trt = new Namespace(TensorRTRegistry.NAMESPACE);
            namespace.addNamespace(trt);
            new TensorRTRegistry(this).registerTensorRTFunctions(trt);
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

    public boolean isEnableMultiGPU() {
        return enableMultiGPU;
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

    private static ExecutionPolicyEnum parseExecutionPolicy(String policyString) {
        if (policyString.equals(ExecutionPolicyEnum.SYNC.getName())) return ExecutionPolicyEnum.SYNC;
        else if (policyString.equals(ExecutionPolicyEnum.ASYNC.getName())) return ExecutionPolicyEnum.ASYNC;
        else {
            LOGGER.warning("unknown execution policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_EXECUTION_POLICY);
            return GrCUDAContext.DEFAULT_EXECUTION_POLICY;
        }
    }

    private static DependencyPolicyEnum parseDependencyPolicy(String policyString) {
        if (policyString.equals(DependencyPolicyEnum.WITH_CONST.getName())) return DependencyPolicyEnum.WITH_CONST;
        else if (policyString.equals(DependencyPolicyEnum.NO_CONST.getName())) return DependencyPolicyEnum.NO_CONST;
        else {
            LOGGER.warning("Warning: unknown dependency policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_DEPENDENCY_POLICY);
            return GrCUDAContext.DEFAULT_DEPENDENCY_POLICY;
        }
    }

    private static RetrieveNewStreamPolicyEnum parseRetrieveStreamPolicy(String policyString) {
        if (policyString.equals(RetrieveNewStreamPolicyEnum.FIFO.getName())) return RetrieveNewStreamPolicyEnum.FIFO;
        else if (policyString.equals(RetrieveNewStreamPolicyEnum.ALWAYS_NEW.getName())) return RetrieveNewStreamPolicyEnum.ALWAYS_NEW;
        else {
            LOGGER.warning("Warning: unknown new stream retrieval policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_RETRIEVE_STREAM_POLICY);
            return GrCUDAContext.DEFAULT_RETRIEVE_STREAM_POLICY;
        }
    }

    private static RetrieveParentStreamPolicyEnum parseParentStreamPolicy(String policyString) {
        if (Objects.equals(policyString, RetrieveParentStreamPolicyEnum.DISJOINT.getName())) return RetrieveParentStreamPolicyEnum.DISJOINT;
        else if (Objects.equals(policyString, RetrieveParentStreamPolicyEnum.SAME_AS_PARENT.getName())) return RetrieveParentStreamPolicyEnum.SAME_AS_PARENT;
        else {
            LOGGER.warning("Warning: unknown parent stream retrieval policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_PARENT_STREAM_POLICY);
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
