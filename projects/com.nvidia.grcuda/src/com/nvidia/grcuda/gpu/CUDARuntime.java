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
package com.nvidia.grcuda.gpu;

import static com.nvidia.grcuda.functions.Function.checkArgumentLength;
import static com.nvidia.grcuda.functions.Function.expectInt;
import static com.nvidia.grcuda.functions.Function.expectLong;
import static com.nvidia.grcuda.functions.Function.expectPositiveLong;

import java.util.HashMap;
import org.graalvm.collections.Pair;
import com.nvidia.grcuda.GPUPointer;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.Namespace;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.functions.CUDAFunction;
import com.nvidia.grcuda.gpu.UnsafeHelper.Integer32Object;
import com.nvidia.grcuda.gpu.UnsafeHelper.Integer64Object;
import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.TruffleLanguage.Env;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.source.Source;

public final class CUDARuntime {

    public static final String CUDA_RUNTIME_LIBRARY_NAME = "cudart";
    public static final String CUDA_LIBRARY_NAME = "cuda";
    static final String NVRTC_LIBRARY_NAME = "nvrtc";

    private final GrCUDAContext context;
    private final NVRuntimeCompiler nvrtc;

    /**
     * Map from library-path to NFI library.
     */
    private final HashMap<String, TruffleObject> loadedLibraries = new HashMap<>();

    /**
     * Map of (library-path, symbol-name) to callable.
     */
    private final HashMap<Pair<String, String>, Object> boundFunctions = new HashMap<>();

    public CUDARuntime(GrCUDAContext context, Env env) {
        this.context = context;
        try {
            TruffleObject libcudart = (TruffleObject) env.parseInternal(
                            Source.newBuilder("nfi", "load " + "lib" + CUDA_RUNTIME_LIBRARY_NAME + ".so", "cudaruntime").build()).call();
            TruffleObject libcuda = (TruffleObject) env.parseInternal(
                            Source.newBuilder("nfi", "load " + "lib" + CUDA_LIBRARY_NAME + ".so", "cuda").build()).call();
            TruffleObject libnvrtc = (TruffleObject) env.parseInternal(
                            Source.newBuilder("nfi", "load " + "lib" + NVRTC_LIBRARY_NAME + ".so", "nvrtc").build()).call();
            loadedLibraries.put(CUDA_RUNTIME_LIBRARY_NAME, libcudart);
            loadedLibraries.put(CUDA_LIBRARY_NAME, libcuda);
            loadedLibraries.put(NVRTC_LIBRARY_NAME, libnvrtc);
        } catch (UnsatisfiedLinkError e) {
            throw new GrCUDAException(e.getMessage());
        }

        nvrtc = new NVRuntimeCompiler(this);
        context.addDisposable(this::shutdown);
    }

    // using this slow/uncached instance since all calls are non-critical
    private static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    interface CallSupport {
        String getName();

        Object getSymbol(CUDARuntime runtime) throws UnknownIdentifierException;

        default void callSymbol(CUDARuntime runtime, Object... arguments) throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
            CompilerAsserts.neverPartOfCompilation();
            Object result = INTEROP.execute(getSymbol(runtime), arguments);
            runtime.checkCUDAReturnCode(result, getName());
        }
    }

    @TruffleBoundary
    public GPUPointer cudaMalloc(long numBytes) {
        try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
            Object callable = CUDARuntimeFunction.CUDA_MALLOC.getSymbol(this);
            Object result = INTEROP.execute(callable, outPointer.getAddress(), numBytes);
            checkCUDAReturnCode(result, "cudaMalloc");
            long addressAllocatedMemory = outPointer.getValueOfPointer();
            return new GPUPointer(addressAllocatedMemory);
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public LittleEndianNativeArrayView cudaMallocManaged(long numBytes) {
        final int cudaMemAttachGlobal = 0x01;
        try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
            Object callable = CUDARuntimeFunction.CUDA_MALLOCMANAGED.getSymbol(this);
            Object result = INTEROP.execute(callable, outPointer.getAddress(), numBytes, cudaMemAttachGlobal);
            checkCUDAReturnCode(result, "cudaMallocManaged");
            long addressAllocatedMemory = outPointer.getValueOfPointer();
            return new LittleEndianNativeArrayView(addressAllocatedMemory, numBytes);
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cudaFree(LittleEndianNativeArrayView memory) {
        try {
            Object callable = CUDARuntimeFunction.CUDA_FREE.getSymbol(this);
            Object result = INTEROP.execute(callable, memory.getStartAddress());
            checkCUDAReturnCode(result, "cudaFree");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cudaFree(GPUPointer pointer) {
        try {
            Object callable = CUDARuntimeFunction.CUDA_FREE.getSymbol(this);
            Object result = INTEROP.execute(callable, pointer.getRawPointer());
            checkCUDAReturnCode(result, "cudaFree");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cudaDeviceSynchronize() {
        try {
            Object callable = CUDARuntimeFunction.CUDA_DEVICESYNCHRONIZE.getSymbol(this);
            Object result = INTEROP.execute(callable);
            checkCUDAReturnCode(result, "cudaDeviceSynchronize");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cudaMemcpy(long destPointer, long fromPointer, long numBytesToCopy) {
        try {
            Object callable = CUDARuntimeFunction.CUDA_MEMCPY.getSymbol(this);
            if (numBytesToCopy < 0) {
                throw new IllegalArgumentException("requested negative number of bytes to copy " + numBytesToCopy);
            }
            // cudaMemcpyKind from driver_types.h (default: direction of transfer is inferred
            // from the pointer values, uses virtual addressing)
            final long cudaMemcpyDefault = 4;
            Object result = INTEROP.execute(callable, destPointer, fromPointer, numBytesToCopy, cudaMemcpyDefault);
            checkCUDAReturnCode(result, "cudaMemcpy");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public DeviceMemoryInfo cudaMemGetInfo() {
        final String symbol = "cudaMemGetInfo";
        final String nfiSignature = "(pointer, pointer): sint32";
        try (Integer64Object freeBytes = UnsafeHelper.createInteger64Object();
                        Integer64Object totalBytes = UnsafeHelper.createInteger64Object()) {
            Object callable = getSymbol(CUDA_RUNTIME_LIBRARY_NAME, symbol, nfiSignature);
            Object result = INTEROP.execute(callable, freeBytes.getAddress(), totalBytes.getAddress());
            checkCUDAReturnCode(result, symbol);
            return new DeviceMemoryInfo(freeBytes.getValue(), totalBytes.getValue());
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cudaDeviceReset() {
        try {
            Object callable = CUDARuntimeFunction.CUDA_DEVICERESET.getSymbol(this);
            Object result = INTEROP.execute(callable);
            checkCUDAReturnCode(result, "cudaDeviceReset");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public int cudaGetDeviceCount() {
        try (UnsafeHelper.Integer32Object deviceCount = UnsafeHelper.createInteger32Object()) {
            Object callable = CUDARuntimeFunction.CUDA_GETDEVICECOUNT.getSymbol(this);
            Object result = INTEROP.execute(callable, deviceCount.getAddress());
            checkCUDAReturnCode(result, "cudaGetDeviceCount");
            return deviceCount.getValue();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cudaSetDevice(int device) {
        try {
            Object callable = CUDARuntimeFunction.CUDA_SETDEVICE.getSymbol(this);
            Object result = INTEROP.execute(callable, device);
            checkCUDAReturnCode(result, "cudaSetDevice");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public int cudaGetDevice() {
        try (Integer32Object deviceId = UnsafeHelper.createInteger32Object()) {
            Object callable = CUDARuntimeFunction.CUDA_GETDEVICE.getSymbol(this);
            Object result = INTEROP.execute(callable, deviceId.getAddress());
            checkCUDAReturnCode(result, "cudaGetDevice");
            return deviceId.getValue();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public int cudaDeviceGetAttribute(CUDADeviceAttribute attribute, int deviceId) {
        try (Integer32Object value = UnsafeHelper.createInteger32Object()) {
            Object callable = CUDARuntimeFunction.CUDA_DEVICEGETATTRIBUTE.getSymbol(this);
            Object result = INTEROP.execute(callable, value.getAddress(), attribute.getAttributeCode(), deviceId);
            checkCUDAReturnCode(result, "cudaDeviceGetAttribute");
            return value.getValue();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public Object getDeviceName(int deviceOrdinal) {
        return cuDeviceGetName(cuDeviceGet(deviceOrdinal));
    }

    @TruffleBoundary
    public String cudaGetErrorString(int errorCode) {
        try {
            Object callable = CUDARuntimeFunction.CUDA_GETERRORSTRING.getSymbol(this);
            Object result = INTEROP.execute(callable, errorCode);
            return INTEROP.asString(result);
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    /**
     * Get function as callable from native library.
     *
     * @param libraryPath path to library (.so file)
     * @param symbolName name of the function (symbol) too look up
     * @param signature NFI signature of the function
     * @return a callable as a TruffleObject
     */
    @TruffleBoundary
    public Object getSymbol(String libraryPath, String symbolName, String signature) throws UnknownIdentifierException {
        return getSymbol(libraryPath, symbolName, signature, "");
    }

    @TruffleBoundary
    public Object getSymbol(String libraryPath, String symbolName, String signature, String hint) throws UnknownIdentifierException {

        Pair<String, String> functionKey = Pair.create(libraryPath, symbolName);
        Object callable = boundFunctions.get(functionKey);
        if (callable == null) {
            // symbol does not exist or not yet bound
            TruffleObject library = loadedLibraries.get(libraryPath);
            if (library == null) {
                try {
                    // library does not exist or is not loaded yet
                    library = (TruffleObject) context.getEnv().parseInternal(
                                    Source.newBuilder("nfi", "load \"" + libraryPath + "\"", libraryPath).build()).call();
                } catch (UnsatisfiedLinkError e) {
                    throw new GrCUDAException("unable to load shared library '" + libraryPath + "': " + e.getMessage() + hint);
                }

                loadedLibraries.put(libraryPath, library);
            }
            try {
                Object symbol = INTEROP.readMember(library, symbolName);
                callable = INTEROP.invokeMember(symbol, "bind", signature);
            } catch (UnsatisfiedLinkError | UnsupportedMessageException | ArityException | UnsupportedTypeException e) {
                throw new GrCUDAException("unexpected behavior: " + e.getMessage());
            }
            boundFunctions.put(functionKey, callable);
        }
        return callable;
    }

    private void checkCUDAReturnCode(Object result, String... function) {
        if (!(result instanceof Integer)) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("expected return code as Integer object in " + GrCUDAException.format(function) + ", got " + result.getClass().getName());
        }
        Integer returnCode = (Integer) result;
        if (returnCode != 0) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(returnCode, cudaGetErrorString(returnCode), function);
        }
    }

    public void registerCUDAFunctions(Namespace rootNamespace) {
        for (CUDARuntimeFunction function : CUDARuntimeFunction.values()) {
            rootNamespace.addFunction(new CUDAFunction(function, this));
        }
    }

    public enum CUDARuntimeFunction implements CUDAFunction.Spec, CallSupport {
        CUDA_DEVICEGETATTRIBUTE("cudaDeviceGetAttribute", "(pointer, sint32, sint32): sint32") {
            @Override
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 2);
                int attributeCode = expectInt(args[0]);
                int deviceId = expectInt(args[1]);
                try (UnsafeHelper.Integer32Object value = UnsafeHelper.createInteger32Object()) {
                    callSymbol(cudaRuntime, value.getAddress(), attributeCode, deviceId);
                    return value.getValue();
                }
            }
        },
        CUDA_DEVICERESET("cudaDeviceReset", "(): sint32") {
            @Override
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, InteropException {
                checkArgumentLength(args, 0);
                callSymbol(cudaRuntime);
                return NoneValue.get();
            }
        },
        CUDA_DEVICESYNCHRONIZE("cudaDeviceSynchronize", "(): sint32") {
            @Override
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, InteropException, UnsupportedMessageException {
                checkArgumentLength(args, 0);
                callSymbol(cudaRuntime);
                return NoneValue.get();
            }
        },
        CUDA_FREE("cudaFree", "(pointer): sint32") {
            @Override
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 1);
                Object pointerObj = args[0];
                long addr;
                if (pointerObj instanceof GPUPointer) {
                    addr = ((GPUPointer) pointerObj).getRawPointer();
                } else if (pointerObj instanceof LittleEndianNativeArrayView) {
                    addr = ((LittleEndianNativeArrayView) pointerObj).getStartAddress();
                } else {
                    throw new GrCUDAException("expected GPUPointer or LittleEndianNativeArrayView");
                }
                callSymbol(cudaRuntime, addr);
                return NoneValue.get();
            }
        },
        CUDA_GETDEVICE("cudaGetDevice", "(pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, InteropException {
                checkArgumentLength(args, 0);
                try (UnsafeHelper.Integer32Object deviceId = UnsafeHelper.createInteger32Object()) {
                    callSymbol(cudaRuntime, deviceId.getAddress());
                    return deviceId.getValue();
                }
            }
        },
        CUDA_GETDEVICECOUNT("cudaGetDeviceCount", "(pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, InteropException {
                checkArgumentLength(args, 0);
                try (UnsafeHelper.Integer32Object deviceCount = UnsafeHelper.createInteger32Object()) {
                    callSymbol(cudaRuntime, deviceCount.getAddress());
                    return deviceCount.getValue();
                }
            }
        },
        CUDA_GETERRORSTRING("cudaGetErrorString", "(sint32): string") {
            @Override
            @TruffleBoundary
            public String call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 1);
                int errorCode = expectInt(args[0]);
                Object result = INTEROP.execute(getSymbol(cudaRuntime), errorCode);
                return INTEROP.asString(result);
            }
        },
        CUDA_MALLOC("cudaMalloc", "(pointer, uint64): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 1);
                long numBytes = expectLong(args[0]);
                try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
                    callSymbol(cudaRuntime, outPointer.getAddress(), numBytes);
                    long addressAllocatedMemory = outPointer.getValueOfPointer();
                    return new GPUPointer(addressAllocatedMemory);
                }
            }
        },
        CUDA_MALLOCMANAGED("cudaMallocManaged", "(pointer, uint64, sint32): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 1);
                final int cudaMemAttachGlobal = 0x01;
                long numBytes = expectLong(args[0]);
                try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
                    callSymbol(cudaRuntime, outPointer.getAddress(), numBytes, cudaMemAttachGlobal);
                    long addressAllocatedMemory = outPointer.getValueOfPointer();
                    return new GPUPointer(addressAllocatedMemory);
                }
            }
        },
        CUDA_SETDEVICE("cudaSetDevice", "(sint32): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 1);
                int device = expectInt(args[0]);
                callSymbol(cudaRuntime, device);
                return NoneValue.get();
            }
        },
        CUDA_MEMCPY("cudaMemcpy", "(pointer, pointer, uint64, sint32): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 3);
                long destPointer = expectLong(args[0]);
                long fromPointer = expectLong(args[1]);
                long numBytesToCopy = expectPositiveLong(args[2]);
                // cudaMemcpyKind from driver_types.h (default: direction of transfer is
                // inferred from the pointer values, uses virtual addressing)
                final long cudaMemcpyDefault = 4;
                callSymbol(cudaRuntime, destPointer, fromPointer, numBytesToCopy, cudaMemcpyDefault);
                return NoneValue.get();
            }
        };

        private final String name;
        private final String nfiSignature;

        CUDARuntimeFunction(String name, String nfiSignature) {
            this.name = name;
            this.nfiSignature = nfiSignature;
        }

        public String getName() {
            return name;
        }

        public Object getSymbol(CUDARuntime runtime) throws UnknownIdentifierException {
            return runtime.getSymbol(CUDA_RUNTIME_LIBRARY_NAME, name, nfiSignature);
        }
    }

    private HashMap<String, CUModule> loadedModules = new HashMap<>();

    @TruffleBoundary
    public Kernel loadKernel(String cubinFile, String kernelName, String signature) {
        CUModule module = loadedModules.get(cubinFile);
        try {
            if (module == null) {
                module = cuModuleLoad(cubinFile);
            }
            long kernelFunction = cuModuleGetFunction(module, kernelName);
            return new Kernel(this, kernelName, module, kernelFunction, signature);
        } catch (Exception e) {
            if ((module != null) && (module.getRefCount() == 1)) {
                cuModuleUnload(module);
            }
            throw e;
        }
    }

    @TruffleBoundary
    public Kernel buildKernel(String code, String kernelName, String signature) {
        String moduleName = "truffle" + context.getNextModuleId();
        PTXKernel ptx = nvrtc.compileKernel(code, kernelName, moduleName, "--std=c++14");
        CUModule module = null;
        try {
            module = cuModuleLoadData(ptx.getPtxSource(), moduleName);
            long kernelFunction = cuModuleGetFunction(module, ptx.getLoweredKernelName());
            return new Kernel(this, ptx.getLoweredKernelName(), module, kernelFunction,
                            signature, ptx.getPtxSource());
        } catch (Exception e) {
            if (module != null) {
                cuModuleUnload(module);
            }
            throw e;
        }
    }

    @TruffleBoundary
    public CUModule cuModuleLoad(String cubinName) {
        assertCUDAInitialized();
        if (loadedModules.containsKey(cubinName)) {
            throw new GrCUDAException("A module for " + cubinName + " was already loaded.");
        }
        try (UnsafeHelper.Integer64Object modulePtr = UnsafeHelper.createInteger64Object()) {
            Object callable = CUDADriverFunction.CU_MODULELOAD.getSymbol(this);
            Object result = INTEROP.execute(callable,
                            modulePtr.getAddress(), cubinName);
            checkCUReturnCode(result, "cuModuleLoad");
            CUModule module = new CUModule(cubinName, modulePtr.getValue());
            loadedModules.put(cubinName, module);
            return module;
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public CUModule cuModuleLoadData(String ptx, String moduleName) {
        assertCUDAInitialized();
        if (loadedModules.containsKey(moduleName)) {
            throw new GrCUDAException("A module for " + moduleName + " was already loaded.");
        }
        try (UnsafeHelper.Integer64Object modulePtr = UnsafeHelper.createInteger64Object()) {
            Object callable = CUDADriverFunction.CU_MODULELOADDATA.getSymbol(this);
            Object result = INTEROP.execute(callable,
                            modulePtr.getAddress(), ptx);
            checkCUReturnCode(result, "cuModuleLoadData");
            CUModule module = new CUModule(moduleName, modulePtr.getValue());
            loadedModules.put(moduleName, module);
            return module;
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cuModuleUnload(CUModule module) {
        try {
            Object callable = CUDADriverFunction.CU_MODULEUNLOAD.getSymbol(this);
            Object result = INTEROP.execute(callable, module.module);
            checkCUReturnCode(result, "cuModuleUnload");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public long cuModuleGetFunction(CUModule kernelModule, String kernelName) {
        try (UnsafeHelper.Integer64Object functionPtr = UnsafeHelper.createInteger64Object()) {
            Object callable = CUDADriverFunction.CU_MODULEGETFUNCTION.getSymbol(this);
            Object result = INTEROP.execute(callable,
                            functionPtr.getAddress(), kernelModule.module, kernelName);
            checkCUReturnCode(result, "cuModuleGetFunction");
            return functionPtr.getValue();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cuCtxSynchronize() {
        assertCUDAInitialized();
        try {
            Object callable = CUDADriverFunction.CU_CTXSYNCHRONIZE.getSymbol(this);
            Object result = INTEROP.execute(callable);
            checkCUReturnCode(result, "cuCtxSynchronize");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cuLaunchKernel(Kernel kernel, KernelConfig config, KernelArguments args) {
        try {
            Object callable = CUDADriverFunction.CU_LAUNCHKERNEL.getSymbol(this);
            Dim3 gridSize = config.getGridSize();
            Dim3 blockSize = config.getBlockSize();
            Object result = INTEROP.execute(callable,
                            kernel.getKernelFunction(),
                            gridSize.getX(),
                            gridSize.getY(),
                            gridSize.getZ(),
                            blockSize.getX(),
                            blockSize.getY(),
                            blockSize.getZ(),
                            config.getDynamicSharedMemoryBytes(),
                            config.getStream(),
                            args.getPointer(),              // pointer to kernel arguments array
                            0                               // extra args
            );
            checkCUReturnCode(result, "cuLaunchKernel");
            cudaDeviceSynchronize();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    private void cuInit() {
        try {
            Object callable = CUDADriverFunction.CU_INIT.getSymbol(this);
            int flags = 0; // must be zero as per CUDA Driver API documentation
            Object result = INTEROP.execute(callable, flags);
            checkCUReturnCode(result, "cuInit");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    private int cuDeviceGetCount() {
        try (UnsafeHelper.Integer32Object devCount = UnsafeHelper.createInteger32Object()) {
            Object callable = CUDADriverFunction.CU_DEVICEGETCOUNT.getSymbol(this);
            Object result = INTEROP.execute(callable, devCount.getAddress());
            checkCUReturnCode(result, "cuDeviceGetCount");
            return devCount.getValue();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    private int cuDeviceGet(int deviceOrdinal) {
        assertCUDAInitialized();
        try (UnsafeHelper.Integer32Object deviceObj = UnsafeHelper.createInteger32Object()) {
            Object callable = CUDADriverFunction.CU_DEVICEGET.getSymbol(this);
            Object result = INTEROP.execute(callable, deviceObj.getAddress(), deviceOrdinal);
            checkCUReturnCode(result, "cuDeviceGet");
            return deviceObj.getValue();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    private String cuDeviceGetName(int cuDeviceId) {
        final int maxLength = 256;
        try (UnsafeHelper.StringObject nameString = new UnsafeHelper.StringObject(maxLength)) {
            Object callable = CUDADriverFunction.CU_DEVICEGETNAME.getSymbol(this);
            Object result = INTEROP.execute(callable, nameString.getAddress(), maxLength, cuDeviceId);
            checkCUReturnCode(result, "cuDeviceGetName");
            return nameString.getZeroTerminatedString();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    private long cuCtxCreate(int flags, int cudevice) {
        try (UnsafeHelper.PointerObject pctx = UnsafeHelper.createPointerObject()) {
            Object callable = CUDADriverFunction.CU_CTXCREATE.getSymbol(this);
            Object result = INTEROP.execute(callable, pctx.getAddress(), flags, cudevice);
            checkCUReturnCode(result, "cuCtxCreate");
            return pctx.getValueOfPointer();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    private long cuDevicePrimaryCtxRetain(int cudevice) {
        try (UnsafeHelper.PointerObject pctx = UnsafeHelper.createPointerObject()) {
            Object callable = CUDADriverFunction.CU_DEVICEPRIMARYCTXRETAIN.getSymbol(this);
            Object result = INTEROP.execute(callable, pctx.getAddress(), cudevice);
            checkCUReturnCode(result, "cuDevicePrimaryCtxRetain");
            return pctx.getValueOfPointer();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    private void cuCtxDestroy(long ctx) {
        try {
            Object callable = CUDADriverFunction.CU_CTXCREATE.getSymbol(this);
            Object result = INTEROP.execute(callable, ctx);
            checkCUReturnCode(result, "cuCtxDestroy");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    private void assertCUDAInitialized() {
        if (!context.isCUDAInitialized()) {
            // a simple way to create the device context in the driver is to call CUDA function
            cudaDeviceSynchronize();
            context.setCUDAInitialized();
        }
    }

    @SuppressWarnings("static-method")
    private static void checkCUReturnCode(Object result, String... function) {
        int returnCode;
        try {
            returnCode = INTEROP.asInt(result);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(
                            "expected return code as Integer object in " + function + ", got " +
                                            result.getClass().getName());
        }
        if (returnCode != 0) {
            throw new GrCUDAException(returnCode, DriverAPIErrorMessages.getString(returnCode), function);
        }
    }

    private void shutdown() {
        // unload all modules
        for (CUModule module : loadedModules.values()) {
            try {
                cuModuleUnload(module);
            } catch (Exception e) {
                /* ignore exception */
            }
        }
    }

    public enum CUDADriverFunction {
        CU_CTXCREATE("cuCtxCreate", "(pointer, uint32, sint32) :sint32"),
        CU_CTXDESTROY("cuCtxDestroy", "(pointer): sint32"),
        CU_CTXSYNCHRONIZE("cuCtxSynchronize", "(): sint32"),
        CU_DEVICEGETCOUNT("cuDeviceGetCount", "(pointer): sint32"),
        CU_DEVICEGET("cuDeviceGet", "(pointer, sint32): sint32"),
        CU_DEVICEGETNAME("cuDeviceGetName", "(pointer, sint32, sint32): sint32"),
        CU_DEVICEPRIMARYCTXRETAIN("cuDevicePrimaryCtxRetain", "(pointer, sint32): sint32"),
        CU_INIT("cuInit", "(uint32): sint32"),
        CU_LAUNCHKERNEL("cuLaunchKernel", "(uint64, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uint64, pointer, pointer): sint32"),
        CU_MODULELOAD("cuModuleLoad", "(pointer, string): sint32"),
        CU_MODULELOADDATA("cuModuleLoadData", "(pointer, string): sint32"),
        CU_MODULEUNLOAD("cuModuleUnload", "(uint64): sint32"),
        CU_MODULEGETFUNCTION("cuModuleGetFunction", "(pointer, uint64, string): sint32");

        private final String symbolName;
        private final String signature;

        CUDADriverFunction(String symbolName, String nfiSignature) {
            this.symbolName = symbolName;
            this.signature = nfiSignature;
        }

        public Object getSymbol(CUDARuntime runtime) throws UnknownIdentifierException {
            return runtime.getSymbol(CUDA_LIBRARY_NAME, symbolName, signature);
        }
    }

    /** CUDA device attributes from driver_types.h CUDA header. */
    public enum CUDADeviceAttribute {
        MAX_THREADS_PER_BLOCK("maxThreadsPerBlock", 1),
        MAX_BLOCK_DIMX("maxBlockDimX", 2),
        MAX_BLOCK_DIMY("maxBlockDimY", 3),
        MAX_BLOCK_DIMZ("maxBlockDimZ", 4),
        MAX_GRID_DIMX("maxGridDimX", 5),
        MAX_GRID_DIMY("maxGridDimY", 6),
        MAX_GRID_DIMZ("maxGridDimZ", 7),
        MAX_SHARED_MEMORY_PER_BLOCK("maxSharedMemoryPerBlock", 8),
        TOTAL_CONSTANT_MEMORY("totalConstantMemory", 9),
        WARPSIZE("warpSize", 10),
        MAX_PITCH("maxPitch", 11),
        MAX_REGISTERS_PER_BLOCK("maxRegistersPerBlock", 12),
        CLOCK_RATE("clockRate", 13),
        TEXTURE_ALIGNMENT("textureAlignment", 14),
        GPU_OVERLAP("gpuOverlap", 15),
        MULTI_PROCESSOR_COUNT("multiProcessorCount", 16),
        KERNEL_EXEC_TIMEOUT("kernelExecTimeout", 17),
        INTEGRATED("integrated", 18),
        CAN_MAP_HOST_MEMORY("canMapHostMemory", 19),
        COMPUTE_MODE("computeMode", 20),
        MAX_TEXTURE1D_WIDTH("maxTexture1DWidth", 21),
        MAX_TEXTURE2D_WIDTH("maxTexture2DWidth", 22),
        MAX_TEXTURE2D_HEIGHT("maxTexture2DHeight", 23),
        MAX_TEXTURE3D_WIDTH("maxTexture3DWidth", 24),
        MAX_TEXTURE3D_HEIGHT("maxTexture3DHeight", 25),
        MAX_TEXTURE3D_DEPTH("maxTexture3DDepth", 26),
        MAX_TEXTURE2D_LAYERED_WIDTH("maxTexture2DLayeredWidth", 27),
        MAX_TEXTURE2D_LAYERED_HEIGHT("maxTexture2DLayeredHeight", 28),
        MAX_TEXTURE2D_LAYERED_LAYERS("maxTexture2DLayeredLayers", 29),
        SURFACE_ALIGNMENT("surfaceAlignment", 30),
        CONCURRENT_KERNELS("concurrentKernels", 31),
        ECC_ENABLED("eccEnabled", 32),
        PCI_BUS_ID("pciBusId", 33),
        PCI_DEVICE_ID("pciDeviceId", 34),
        TCC_DRIVER("tccDriver", 35),
        MEMORY_CLOCK_RATE("memoryClockRate", 36),
        GLOBAL_MEMORY_BUS_WIDTH("globalMemoryBusWidth", 37),
        L2_CACHE_SIZE("l2CacheSize", 38),
        MAX_THREADS_PER_MULTIPROCESSOR("maxThreadsPerMultiProcessor", 39),
        ASYNC_ENGINE_COUNT("asyncEngineCount", 40),
        UNIFIED_ADDRESSING("unifiedAddressing", 41),
        MAX_TEXTURE1D_LAYERED_WIDTH("maxTexture1DLayeredWidth", 42),
        MAX_TEXTURE1D_LAYERED_LAYERS("maxTexture1DLayeredLayers", 43),
        MAX_TEXTURE2D_GATHER_WIDTH("maxTexture2DGatherWidth", 45),
        MAX_TEXTURE2D_GATHER_HEIGHT("maxTexture2DGatherHeight", 46),
        MAX_TEXTURE3D_WIDTH_ALT("maxTexture3DWidthAlt", 47),
        MAX_TEXTURE3D_HEIGHT_ALT("maxTexture3DHeightAlt", 48),
        MAX_TEXTURE3D_DEPTH_ALT("maxTexture3DDepthAlt", 49),
        PCI_DOMAIN_ID("pciDomainId", 50),
        TEXTURE_PITCH_ALIGNMENT("texturePitchAlignment", 51),
        MAX_TEXTURE_CUBEMAP_WIDTH("maxTextureCubemapWidth", 52),
        MAX_TEXTURE_CUBEMAP_LAYERED_WIDTH("maxTextureCubemapLayeredWidth", 53),
        MAX_TEXTURE_CUBEMAP_LAYERED_LAYERS("maxTextureCubemapLayeredLayers", 54),
        MAX_SURFACE1D_WIDTH("maxSurface1DWidth", 55),
        MAX_SURFACE2D_WIDTH("maxSurface2DWidth", 56),
        MAX_SURFACE2D_HEIGHT("maxSurface2DHeight", 57),
        MAX_SURFACE3D_WIDTH("maxSurface3DWidth", 58),
        MAX_SURFACE3D_HEIGHT("maxSurface3DHeight", 59),
        MAX_SURFACE3D_DEPTH("maxSurface3DDepth", 60),
        MAX_SURFACE1D_LAYERED_WIDTH("maxSurface1DLayeredWidth", 61),
        MAX_SURFACE1D_LAYERED_LAYERS("maxSurface1DLayeredLayers", 62),
        MAX_SURFACE2D_LAYERED_WIDTH("maxSurface2DLayeredWidth", 63),
        MAX_SURFACE2D_LAYERED_HEIGHT("maxSurface2DLayeredHeight", 64),
        MAX_SURFACE2D_LAYERED_LAYERS("maxSurface2DLayeredLayers", 65),
        MAX_SURFACE_CUBEMAP_WIDTH("maxSurfaceCubemapWidth", 66),
        MAX_SURFACE_CUBEMAP_LAYERED_WIDTH("maxSurfaceCubemapLayeredWidth", 67),
        MAX_SURFACE_CUBEMAP_LAYERED_LAYERS("maxSurfaceCubemapLayeredLayers", 68),
        MAX_TEXTURE1D_LINEAR_WIDTH("maxTexture1DLinearWidth", 69),
        MAX_TEXTURE2D_LINEAR_WIDTH("maxTexture2DLinearWidth", 70),
        MAX_TEXTURE2D_LINEAR_HEIGHT("maxTexture2DLinearHeight", 71),
        MAX_TEXTURE2D_LINEAR_PITCH("maxTexture2DLinearPitch", 72),
        MAX_TEXTURE2D_MIPMAPPED_WIDTH("maxTexture2DMipmappedWidth", 73),
        MAX_TEXTURE2D_MIPMAPPED_HEIGHT("maxTexture2DMipmappedHeight", 74),
        COMPUTE_CAPABILITY_MAJOR("computeCapabilityMajor", 75),
        COMPUTE_CAPABILITY_MINOR("computeCapabilityMinor", 76),
        MAX_TEXTURE1D_MIPMAPPED_WIDTH("maxTexture1DMipmappedWidth", 77),
        STREAM_PRIORITIES_SUPPORTED("streamPrioritiesSupported", 78),
        GLOBAL_L1_CACHE_SUPPORTED("globalL1CacheSupported", 79),
        LOCAL_L1_CACHE_SUPPORTED("localL1CacheSupported", 80),
        MAX_SHARED_MEMORY_PER_MULTIPROCESSOR("maxSharedMemoryPerMultiprocessor", 81),
        MAX_REGISTERS_PER_MULTIPROCESSOR("maxRegistersPerMultiprocessor", 82),
        MANAGED_MEMORY("managedMemory", 83),
        IS_MULTI_GPU_BOARD("isMultiGpuBoard", 84),
        MULTI_GPU_BOARD_GROUP_ID("multiGpuBoardGroupID", 85),
        HOST_NATIVE_ATOMIC_SUPPORTED("hostNativeAtomicSupported", 86),
        SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO("singleToDoublePrecisionPerfRatio", 87),
        PAGEABLE_MEMORY_ACCESS("pageableMemoryAccess", 88),
        CONCURRENT_MANAGED_ACCESS("concurrentManagedAccess", 89),
        COMPUTE_PREEMPTION_SUPPORTED("computePreemptionSupported", 90),
        CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM("canUseHostPointerForRegisteredMem", 91),
        COOPERATIVE_LAUNCH("cooperativeLaunch", 95),
        COOPERATIVE_MULTI_DEVICE_LAUNCH("cooperativeMultiDeviceLaunch", 96),
        MAX_SHARED_MEMORY_PER_BLOCK_OPTIN("maxSharedMemoryPerBlockOptin", 97),
        CAN_FLUSH_REMOTE_WRITES("canFlushRemoteWrites", 98),
        HOST_REGISTER_SUPPORTED("hostRegisterSupported", 99),
        PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES("pageableMemoryAccessUsesHostPageTables", 100),
        DIRECT_MANAGED_MEM_ACCESS_FROM_HOST("directManagedMemAccessFromHost", 101);

        final String attributeName;
        final int attributeCode;

        String getAttributeName() {
            return attributeName;
        }

        int getAttributeCode() {
            return attributeCode;
        }

        CUDADeviceAttribute(String name, int code) {
            this.attributeName = name;
            this.attributeCode = code;
        }
    }

    final class CUModule {
        final String cubinFile;
        final long module;
        int refCount;

        CUModule(String cubinFile, long module) {
            this.cubinFile = cubinFile;
            this.module = module;
            this.refCount = 1;
        }

        public long getModulePointer() {
            return module;
        }

        public int getRefCount() {
            return refCount;
        }

        public void incrementRefCount() {
            refCount += 1;
        }

        public void decrementRefCount() {
            refCount -= 1;
            if (refCount == 0) {
                cuModuleUnload(this);
            }
        }

        @Override
        public boolean equals(Object other) {
            if (other instanceof CUModule) {
                CUModule otherModule = (CUModule) other;
                return otherModule.cubinFile.equals(cubinFile);
            } else {
                return false;
            }
        }

        @Override
        public int hashCode() {
            return cubinFile.hashCode();
        }
    }
}

final class DeviceMemoryInfo {
    private final long freeBytes;
    private final long totalBytes;

    DeviceMemoryInfo(long freeBytes, long totalBytes) {
        this.freeBytes = freeBytes;
        this.totalBytes = totalBytes;
    }

    public long getFreeBytes() {
        return freeBytes;
    }

    public long getTotalBytes() {
        return totalBytes;
    }

    @Override
    public String toString() {
        return String.format("DeviceMemoryInfo(freeBytes=%d bytes, totalBytes=%d bytes", freeBytes, totalBytes);
    }
}
