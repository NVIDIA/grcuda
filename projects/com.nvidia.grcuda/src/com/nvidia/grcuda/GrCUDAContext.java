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

import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.graalvm.options.OptionKey;

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
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.TruffleLanguage.Env;

/**
 * Context for the grCUDA language holds reference to CUDA runtime, a function registry and device
 * resources.
 */
public final class GrCUDAContext {

    private static final String ROOT_NAMESPACE = "CU";

    private final Env env;
    private final CUDARuntime cudaRuntime;
    private final Namespace rootNamespace;
    private final ArrayList<Runnable> disposables = new ArrayList<>();
    private AtomicInteger moduleId = new AtomicInteger(0);
    private volatile boolean cudaInitialized = false;

    // this is used to look up pre-existing call targets for "map" operations, see MapArrayNode
    private final ConcurrentHashMap<Class<?>, CallTarget> uncachedMapCallTargets = new ConcurrentHashMap<>();

    public GrCUDAContext(Env env) {
        this.env = env;
        this.cudaRuntime = new CUDARuntime(this, env);

        Namespace namespace = new Namespace(ROOT_NAMESPACE);
        namespace.addNamespace(namespace);
        namespace.addFunction(new BindFunction());
        namespace.addFunction(new DeviceArrayFunction(cudaRuntime));
        namespace.addFunction(new MapFunction());
        namespace.addFunction(new ShredFunction());
        namespace.addFunction(new BindKernelFunction(cudaRuntime));
        namespace.addFunction(new BuildKernelFunction(cudaRuntime));
        namespace.addFunction(new GetDevicesFunction(cudaRuntime));
        namespace.addFunction(new GetDeviceFunction(cudaRuntime));
        cudaRuntime.registerCUDAFunctions(namespace);
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

    public CUDARuntime getCUDARuntime() {
        return cudaRuntime;
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

    @TruffleBoundary
    public <T> T getOption(OptionKey<T> key) {
        return env.getOptions().get(key);
    }
}
