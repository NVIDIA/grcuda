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
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.cudalibraries.tensorrt.TensorRTRegistry;
import com.nvidia.grcuda.functions.BindAllFunction;
import com.nvidia.grcuda.functions.BindFunction;
import com.nvidia.grcuda.functions.BindKernelFunction;
import com.nvidia.grcuda.functions.BuildKernelFunction;
import com.nvidia.grcuda.functions.DeviceArrayFunction;
import com.nvidia.grcuda.functions.GetDeviceFunction;
import com.nvidia.grcuda.functions.GetDevicesFunction;
import com.nvidia.grcuda.functions.GetOptionsFunction;
import com.nvidia.grcuda.functions.map.MapFunction;
import com.nvidia.grcuda.functions.map.ShredFunction;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.computation.prefetch.PrefetcherEnum;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.executioncontext.SyncGrCUDAExecutionContext;
import com.nvidia.grcuda.cudalibraries.tensorrt.TensorRTRegistry;
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.TruffleLanguage.Env;
import com.oracle.truffle.api.TruffleLogger;

import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Context for the GrCUDA language holds reference to CUDA runtime, a function registry and device
 * resources.
 */
public final class GrCUDAContext {

    private static final String ROOT_NAMESPACE = "CU";

    private static final TruffleLogger LOGGER = GrCUDALogger.getLogger(GrCUDALogger.GRCUDA_LOGGER);

    private final GrCUDAOptionMap grCUDAOptionMap;

    private final Env env;
    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final Namespace rootNamespace;
    private final ArrayList<Runnable> disposables = new ArrayList<>();
    private final AtomicInteger moduleId = new AtomicInteger(0);
    private volatile boolean cudaInitialized = false;

    // this is used to look up pre-existing call targets for "map" operations, see MapArrayNode
    private final ConcurrentHashMap<Class<?>, CallTarget> uncachedMapCallTargets = new ConcurrentHashMap<>();

    public GrCUDAContext(Env env) {
        this.env = env;

        this.grCUDAOptionMap = new GrCUDAOptionMap(env.getOptions());

        // Retrieve the dependency computation policy;
        DependencyPolicyEnum dependencyPolicy = grCUDAOptionMap.getDependencyPolicy();

        // Retrieve the execution policy;
        ExecutionPolicyEnum executionPolicy = grCUDAOptionMap.getExecutionPolicy();

        // FIXME: TensorRT is currently incompatible with the async scheduler. TensorRT is supported in CUDA 11.4, and we cannot test it. 
        //  Once Nvidia adds support for it, we want to remove this limitation;
        if (grCUDAOptionMap.isTensorRTEnabled() && (executionPolicy == ExecutionPolicyEnum.ASYNC)) {
            LOGGER.warning("TensorRT and the asynchronous scheduler are not compatible. Switching to the synchronous scheduler.");
            executionPolicy = ExecutionPolicyEnum.SYNC;
        }

        Boolean inputPrefetch = grCUDAOptionMap.isInputPrefetch();

        // Initialize the execution policy;
        LOGGER.info("using" + executionPolicy.toString() + " execution policy");
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
        namespace.addFunction(new GetOptionsFunction(grCUDAOptionMap));
        this.grCUDAExecutionContext.getCudaRuntime().registerCUDAFunctions(namespace);
        if (grCUDAOptionMap.isCuMLEnabled()) {
            if (this.getCUDARuntime().isArchitectureIsPascalOrNewer()) {
                Namespace ml = new Namespace(CUMLRegistry.NAMESPACE);
                namespace.addNamespace(ml);
                new CUMLRegistry(this).registerCUMLFunctions(ml);
            } else {
                LOGGER.warning("cuML is supported only on GPUs with compute capability >= 6.0 (Pascal and newer). It cannot be enabled.");
            }
        }
        if (grCUDAOptionMap.isCuBLASEnabled()) {
                Namespace blas = new Namespace(CUBLASRegistry.NAMESPACE);
                namespace.addNamespace(blas);
                new CUBLASRegistry(this).registerCUBLASFunctions(blas);
        }
        if (grCUDAOptionMap.isTensorRTEnabled()) {
            Namespace trt = new Namespace(TensorRTRegistry.NAMESPACE);
            namespace.addNamespace(trt);
            new TensorRTRegistry(this).registerTensorRTFunctions(trt);
        }
        if (grCUDAOptionMap.isCuSPARSEEnabled()) {
            Namespace sparse = new Namespace(CUSPARSERegistry.NAMESPACE);
            namespace.addNamespace(sparse);
            new CUSPARSERegistry(this).registerCUSPARSEFunctions(sparse);
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


    /**
     * Compute the maximum number of concurrent threads that can be spawned by GrCUDA.
     * This value is usually smaller or equal than the number of logical CPU threads available on the machine.
     * @return the maximum number of concurrent threads that can be spawned by GrCUDA
     */
    public int getNumberOfThreads() {
        return Runtime.getRuntime().availableProcessors();
    }

    public GrCUDAOptionMap getOptions() {
        return grCUDAOptionMap;
    }

    /**
     * Cleanup the GrCUDA context at the end of the execution;
     */
    public void cleanup() {
        this.grCUDAExecutionContext.cleanup();
    }
}