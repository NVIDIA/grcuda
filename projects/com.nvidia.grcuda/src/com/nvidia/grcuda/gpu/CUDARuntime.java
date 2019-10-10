/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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

import java.util.EnumSet;
import java.util.HashMap;
import org.graalvm.collections.Pair;
import com.nvidia.grcuda.GPUPointer;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.functions.CUDAFunction;
import com.nvidia.grcuda.functions.CUDAFunctionFactory;
import com.nvidia.grcuda.functions.FunctionTable;
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
        TruffleObject libcudart = (TruffleObject) env.parse(
                        Source.newBuilder("nfi", "load " + "lib" + CUDA_RUNTIME_LIBRARY_NAME + ".so", "cudaruntime").build()).call();
        TruffleObject libcuda = (TruffleObject) env.parse(
                        Source.newBuilder("nfi", "load " + "lib" + CUDA_LIBRARY_NAME + ".so", "cuda").build()).call();
        TruffleObject libnvrtc = (TruffleObject) env.parse(
                        Source.newBuilder("nfi", "load " + "lib" + NVRTC_LIBRARY_NAME + ".so", "nvrtc").build()).call();
        loadedLibraries.put(CUDA_RUNTIME_LIBRARY_NAME, libcudart);
        loadedLibraries.put(CUDA_LIBRARY_NAME, libcuda);
        loadedLibraries.put(NVRTC_LIBRARY_NAME, libnvrtc);
        nvrtc = new NVRuntimeCompiler(this);
        context.addDisposable(this::shutdown);
    }

    // using this slow/uncached instance since all calls are non-critical
    private static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    @TruffleBoundary
    public GPUPointer cudaMalloc(long numBytes) {
        try {
            try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
                Object callable = getSymbol(CUDARuntimeFunction.CUDA_MALLOC);
                Object result = INTEROP.execute(callable, outPointer.getAddress(), numBytes);
                checkCUDAReturnCode(result, "cudaMalloc");
                long addressAllocatedMemory = outPointer.getValueOfPointer();
                return new GPUPointer(addressAllocatedMemory);
            }
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public LittleEndianNativeArrayView cudaMallocManaged(long numBytes) {
        final int cudaMemAttachGlobal = 0x01;
        try {
            try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
                Object callable = getSymbol(CUDARuntimeFunction.CUDA_MALLOCMANAGED);
                Object result = INTEROP.execute(callable, outPointer.getAddress(), numBytes, cudaMemAttachGlobal);
                checkCUDAReturnCode(result, "cudaMallocManaged");
                long addressAllocatedMemory = outPointer.getValueOfPointer();
                return new LittleEndianNativeArrayView(addressAllocatedMemory, numBytes);
            }
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public void cudaFree(LittleEndianNativeArrayView memory) {
        try {
            Object callable = getSymbol(CUDARuntimeFunction.CUDA_FREE);
            Object result = INTEROP.execute(callable, memory.getStartAddress());
            checkCUDAReturnCode(result, "cudaFree");
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public void cudaFree(GPUPointer pointer) {
        try {
            Object callable = getSymbol(CUDARuntimeFunction.CUDA_FREE);
            Object result = INTEROP.execute(callable, pointer.getRawPointer());
            checkCUDAReturnCode(result, "cudaFree");
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public void cudaDeviceSynchronize() {
        try {
            Object callable = getSymbol(CUDARuntimeFunction.CUDA_DEVICESYNCHRONIZE);
            Object result = INTEROP.execute(callable);
            checkCUDAReturnCode(result, "cudaDeviceSynchronize");
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public void cudaMemcpy(long destPointer, long fromPointer, long numBytesToCopy) {
        try {
            Object callable = getSymbol(CUDARuntimeFunction.CUDA_MEMCPY);
            if (numBytesToCopy < 0) {
                throw new IllegalArgumentException("requested negative number of bytes to copy " + numBytesToCopy);
            }
            // cudaMemcpyKind from driver_types.h (default: direction of transfer is inferred
            // from the pointer values, uses virtual addressing)
            final long cudaMemcpyDefault = 4;
            Object result = INTEROP.execute(callable, destPointer, fromPointer, numBytesToCopy, cudaMemcpyDefault);
            checkCUDAReturnCode(result, "cudaMemcpy");
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public void cudaDeviceReset() {
        try {
            Object callable = getSymbol(CUDARuntimeFunction.CUDA_DEVICERESET);
            Object result = INTEROP.execute(callable);
            checkCUDAReturnCode(result, "cudaDeviceReset");
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public int cudaGetDeviceCount() {
        try {
            try (UnsafeHelper.Integer32Object deviceCount = UnsafeHelper.createInteger32Object()) {
                Object callable = getSymbol(CUDARuntimeFunction.CUDA_GETDEVICECOUNT);
                Object result = INTEROP.execute(callable, deviceCount.getAddress());
                checkCUDAReturnCode(result, "cudaGetDeviceCount");
                return deviceCount.getValue();
            }
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public void cudaSetDevice(int device) {
        try {
            Object callable = getSymbol(CUDARuntimeFunction.CUDA_SETDEVICE);
            Object result = INTEROP.execute(callable, device);
            checkCUDAReturnCode(result, "cudaSetDevice");
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public String cudaGetErrorString(int errorCode) {
        try {
            Object callable = getSymbol(CUDARuntimeFunction.CUDA_GETERRORSTRING);
            Object result = INTEROP.execute(callable, errorCode);
            return INTEROP.asString(result);
        } catch (InteropException e) {
            throw new RuntimeException(e);
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
        Pair<String, String> functionKey = Pair.create(libraryPath, symbolName);
        Object callable = boundFunctions.get(functionKey);
        if (callable == null) {
            // symbol does not exist or not yet bound
            TruffleObject library = loadedLibraries.get(libraryPath);
            if (library == null) {
                // library does not exist or is not loaded yet
                library = (TruffleObject) context.getEnv().parse(
                                Source.newBuilder("nfi", "load \"" + libraryPath + "\"", libraryPath).build()).call();
                loadedLibraries.put(libraryPath, library);
            }
            Object symbol;
            try {
                symbol = INTEROP.readMember(library, symbolName);
                callable = INTEROP.invokeMember(symbol, "bind", signature);
            } catch (UnsupportedMessageException | ArityException | UnsupportedTypeException e) {
                throw new RuntimeException("unexpected behavior");
            }
            boundFunctions.put(functionKey, callable);
        }
        return callable;
    }

    private Object getSymbol(CUDARuntimeFunction function) throws UnknownIdentifierException {
        return getSymbol(CUDA_RUNTIME_LIBRARY_NAME, function.getFunctionFactory().getName(),
                        function.getFunctionFactory().getNFISignature());
    }

    private void checkCUDAReturnCode(Object result, String function) {
        if (!(result instanceof Integer)) {
            throw new RuntimeException(
                            "expected return code as Integer object in " + function + ", got " +
                                            result.getClass().getName());
        }
        Integer returnCode = (Integer) result;
        if (returnCode != 0) {
            throw new CUDAException(returnCode, cudaGetErrorString(returnCode), function);
        }
    }

    public void registerCUDAFunctions(FunctionTable functionTable) {
        EnumSet.allOf(CUDARuntimeFunction.class).forEach(
                        func -> functionTable.registerFunction(
                                        func.getFunctionFactory().makeFunction(this)));
    }

    public enum CUDARuntimeFunction {
        CUDA_GETDEVICECOUNT(new CUDAFunctionFactory("cudaGetDeviceCount", "", "(pointer): sint32") {
            @Override
            public CUDAFunction makeFunction(CUDARuntime cudaRuntime) {
                return new CUDAFunction(this) {
                    @Override
                    @TruffleBoundary
                    public Object call(Object[] args) throws ArityException {
                        checkArgumentLength(args, 0);
                        try (UnsafeHelper.Integer32Object deviceCount = UnsafeHelper.createInteger32Object()) {
                            Object callable = cudaRuntime.getSymbol(CUDARuntimeFunction.CUDA_GETDEVICECOUNT);
                            Object result = INTEROP.execute(callable, deviceCount.getAddress());
                            cudaRuntime.checkCUDAReturnCode(result, getName());
                            return deviceCount.getValue();
                        } catch (InteropException e) {
                            throw new RuntimeException(e);
                        }
                    }
                };
            }
        }),

        CUDA_DEVICERESET(new CUDAFunctionFactory("cudaDeviceReset", "", "(): sint32") {
            @Override
            public CUDAFunction makeFunction(CUDARuntime cudaRuntime) {
                return new CUDAFunction(this) {
                    @Override
                    @TruffleBoundary
                    public Object call(Object[] args) throws ArityException {
                        checkArgumentLength(args, 0);
                        try {
                            Object callable = cudaRuntime.getSymbol(CUDARuntimeFunction.CUDA_DEVICERESET);
                            Object result = INTEROP.execute(callable);
                            cudaRuntime.checkCUDAReturnCode(result, getName());
                            return 0;  // NFI returns 0 functions that have 'void' as return type
                                       // ???
                        } catch (InteropException e) {
                            throw new RuntimeException(e);
                        }
                    }
                };
            }
        }),
        CUDA_DEVICESYNCHRONIZE(new CUDAFunctionFactory("cudaDeviceSynchronize", "", "(): sint32") {
            @Override
            public CUDAFunction makeFunction(CUDARuntime cudaRuntime) {
                return new CUDAFunction(this) {
                    @Override
                    @TruffleBoundary
                    public Object call(Object[] args) throws ArityException {
                        checkArgumentLength(args, 0);
                        try {
                            Object callable = cudaRuntime.getSymbol(CUDARuntimeFunction.CUDA_DEVICESYNCHRONIZE);
                            Object result = INTEROP.execute(callable);
                            cudaRuntime.checkCUDAReturnCode(result, getName());
                            return 0;  // NFI returns 0 functions that have 'void' as return type
                                       // ???
                        } catch (InteropException e) {
                            throw new RuntimeException(e);
                        }
                    }
                };
            }
        }),
        CUDA_FREE(new CUDAFunctionFactory("cudaFree", "", "(pointer): sint32") {
            @Override
            public CUDAFunction makeFunction(CUDARuntime cudaRuntime) {
                return new CUDAFunction(this) {
                    @Override
                    @TruffleBoundary
                    public Object call(Object[] args) throws ArityException, UnsupportedTypeException {
                        checkArgumentLength(args, 1);
                        Object pointerObj = args[0];
                        long addr;
                        if (pointerObj instanceof GPUPointer) {
                            addr = ((GPUPointer) pointerObj).getRawPointer();
                        } else if (pointerObj instanceof LittleEndianNativeArrayView) {
                            addr = ((LittleEndianNativeArrayView) pointerObj).getStartAddress();
                        } else {
                            throw new RuntimeException("expected GPUPointer or LittleEndianNativeArrayView");
                        }
                        try {
                            Object callable = cudaRuntime.getSymbol(CUDARuntimeFunction.CUDA_FREE);
                            Object result = INTEROP.execute(callable, addr);
                            cudaRuntime.checkCUDAReturnCode(result, getName());
                            return null;
                        } catch (InteropException e) {
                            throw new RuntimeException(e);
                        }
                    }
                };
            }
        }),
        CUDA_GETERRORSTRING(new CUDAFunctionFactory("cudaGetErrorString", "", "(sint32): string") {
            @Override
            public CUDAFunction makeFunction(CUDARuntime cudaRuntime) {
                return new CUDAFunction(this) {
                    @Override
                    @TruffleBoundary
                    public Object call(Object[] args) throws ArityException, UnsupportedTypeException {
                        checkArgumentLength(args, 1);
                        int errorCode = expectInt(args[0]);
                        try {
                            Object callable = cudaRuntime.getSymbol(CUDARuntimeFunction.CUDA_GETERRORSTRING);
                            Object result = INTEROP.execute(callable, errorCode);
                            return INTEROP.asString(result);
                        } catch (InteropException e) {
                            throw new RuntimeException(e);
                        }
                    }
                };
            }
        }),
        CUDA_MALLOC(new CUDAFunctionFactory("cudaMalloc", "", "(pointer, uint64): sint32") {
            @Override
            public CUDAFunction makeFunction(CUDARuntime cudaRuntime) {
                return new CUDAFunction(this) {
                    @Override
                    @TruffleBoundary
                    public Object call(Object[] args) throws ArityException, UnsupportedTypeException {
                        checkArgumentLength(args, 1);
                        long numBytes = expectInt(args[0]);
                        try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
                            Object callable = cudaRuntime.getSymbol(CUDARuntimeFunction.CUDA_MALLOC);
                            Object result = INTEROP.execute(callable, outPointer.getAddress(), numBytes);
                            cudaRuntime.checkCUDAReturnCode(result, getName());
                            long addressAllocatedMemory = outPointer.getValueOfPointer();
                            return new GPUPointer(addressAllocatedMemory);
                        } catch (InteropException e) {
                            throw new RuntimeException(e);
                        }
                    }
                };
            }
        }),
        CUDA_MALLOCMANAGED(
                        new CUDAFunctionFactory("cudaMallocManaged", "", "(pointer, uint64, sint32): sint32") {
                            @Override
                            public CUDAFunction makeFunction(CUDARuntime cudaRuntime) {
                                return new CUDAFunction(this) {
                                    @Override
                                    @TruffleBoundary
                                    public Object call(Object[] args) throws ArityException, UnsupportedTypeException {
                                        checkArgumentLength(args, 1);
                                        final int cudaMemAttachGlobal = 0x01;
                                        long numBytes = expectInt(args[0]);
                                        try (UnsafeHelper.PointerObject outPointer = UnsafeHelper.createPointerObject()) {
                                            Object callable = cudaRuntime.getSymbol(CUDARuntimeFunction.CUDA_MALLOCMANAGED);
                                            Object result = INTEROP.execute(callable, outPointer.getAddress(), numBytes, cudaMemAttachGlobal);
                                            cudaRuntime.checkCUDAReturnCode(result, getName());
                                            long addressAllocatedMemory = outPointer.getValueOfPointer();
                                            return new GPUPointer(addressAllocatedMemory);
                                        } catch (InteropException e) {
                                            throw new RuntimeException(e);
                                        }
                                    }
                                };
                            }
                        }),
        CUDA_SETDEVICE(new CUDAFunctionFactory("cudaSetDevice", "", "(sint32): sint32") {
            @Override
            public CUDAFunction makeFunction(CUDARuntime cudaRuntime) {
                return new CUDAFunction(this) {
                    @Override
                    @TruffleBoundary
                    public Object call(Object[] args) throws ArityException, UnsupportedTypeException {
                        checkArgumentLength(args, 1);
                        int device = expectInt(args[0]);
                        try {
                            Object callable = cudaRuntime.getSymbol(CUDARuntimeFunction.CUDA_SETDEVICE);
                            Object result = INTEROP.execute(callable, device);
                            cudaRuntime.checkCUDAReturnCode(result, getName());
                        } catch (InteropException e) {
                            throw new RuntimeException(e);
                        }
                        return null;
                    }
                };
            }
        }),
        CUDA_MEMCPY(new CUDAFunctionFactory("cudaMemcpy", "", "(pointer, pointer, uint64, sint32): sint32") {
            @Override
            public CUDAFunction makeFunction(CUDARuntime cudaRuntime) {
                return new CUDAFunction(this) {
                    @Override
                    @TruffleBoundary
                    public Object call(Object[] args) throws ArityException, UnsupportedTypeException {
                        checkArgumentLength(args, 3);
                        long destPointer = expectLong(args[0]);
                        long fromPointer = expectLong(args[1]);
                        long numBytesToCopy = expectPositiveLong(args[2]);
                        // cudaMemcpyKind from driver_types.h (default: direction of transfer is
                        // inferred from the pointer values, uses virtual addressing)
                        final long cudaMemcpyDefault = 4;
                        try {
                            Object callable = cudaRuntime.getSymbol(CUDARuntimeFunction.CUDA_MEMCPY);
                            Object result = INTEROP.execute(callable, destPointer, fromPointer, numBytesToCopy, cudaMemcpyDefault);
                            cudaRuntime.checkCUDAReturnCode(result, getName());
                        } catch (InteropException e) {
                            throw new RuntimeException(e);
                        }
                        return null;
                    }
                };
            }
        });

        private final CUDAFunctionFactory factory;

        public CUDAFunctionFactory getFunctionFactory() {
            return factory;
        }

        CUDARuntimeFunction(CUDAFunctionFactory factory) {
            this.factory = factory;
        }

    }

    public Object getSymbol(CUDADriverFunction function) throws UnknownIdentifierException {
        return getSymbol(CUDA_LIBRARY_NAME, function.symbolName, function.signature);
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
            throw new IllegalArgumentException("A module for " + cubinName + " was already loaded.");
        }
        try {
            try (UnsafeHelper.Integer64Object modulePtr = UnsafeHelper.createInteger64Object()) {
                Object callable = getSymbol(CUDADriverFunction.CU_MODULELOAD);
                Object result = INTEROP.execute(callable,
                                modulePtr.getAddress(), cubinName);
                checkCUReturnCode(result, "cuModuleLoad");
                CUModule module = new CUModule(cubinName, modulePtr.getValue());
                loadedModules.put(cubinName, module);
                return module;
            }
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public CUModule cuModuleLoadData(String ptx, String moduleName) {
        assertCUDAInitialized();
        if (loadedModules.containsKey(moduleName)) {
            throw new IllegalArgumentException("A module for " + moduleName + " was already loaded.");
        }
        try {
            try (UnsafeHelper.Integer64Object modulePtr = UnsafeHelper.createInteger64Object()) {
                Object callable = getSymbol(CUDADriverFunction.CU_MODULELOADDATA);
                Object result = INTEROP.execute(callable,
                                modulePtr.getAddress(), ptx);
                checkCUReturnCode(result, "cuModuleLoadData");
                CUModule module = new CUModule(moduleName, modulePtr.getValue());
                loadedModules.put(moduleName, module);
                return module;
            }
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public void cuModuleUnload(CUModule module) {
        try {
            Object callable = getSymbol(CUDADriverFunction.CU_MODULEUNLOAD);
            Object result = INTEROP.execute(callable, module.module);
            checkCUReturnCode(result, "cuModuleUnload");
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public long cuModuleGetFunction(CUModule kernelModule, String kernelName) {
        try {
            try (UnsafeHelper.Integer64Object functionPtr = UnsafeHelper.createInteger64Object()) {
                Object callable = getSymbol(CUDADriverFunction.CU_MODULEGETFUNCTION);
                Object result = INTEROP.execute(callable,
                                functionPtr.getAddress(), kernelModule.module, kernelName);
                checkCUReturnCode(result, "cuModuleGetFunction");
                return functionPtr.getValue();
            }
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public void cuCtxSynchronize() {
        assertCUDAInitialized();
        try {
            Object callable = getSymbol(CUDADriverFunction.CU_CTXSYNCHRONIZE);
            Object result = INTEROP.execute(callable);
            checkCUReturnCode(result, "cuCtxSynchronize");
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    public void cuLaunchKernel(Kernel kernel, KernelConfig config, KernelArguments args) {
        try {
            Object callable = getSymbol(CUDADriverFunction.CU_LAUNCHKERNEL);
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
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    private void cuInit() {
        try {
            Object callable = getSymbol(CUDADriverFunction.CU_INIT);
            int flags = 0; // must be zero as per CUDA Driver API documentation
            Object result = INTEROP.execute(callable, flags);
            checkCUReturnCode(result, "cuInit");
        } catch (InteropException e) {
            throw new RuntimeException(e);
        }
    }

    @TruffleBoundary
    private int cuDeviceGetCount() {
        try (UnsafeHelper.Integer32Object devCount = UnsafeHelper.createInteger32Object()) {
            try {
                Object callable = getSymbol(CUDADriverFunction.CU_DEVICEGETCOUNT);
                Object result = INTEROP.execute(callable, devCount.getAddress());
                checkCUReturnCode(result, "cuDeviceGetCount");
                return devCount.getValue();
            } catch (InteropException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @TruffleBoundary
    private int cuDeviceGet(int deviceOrdinal) {
        try (UnsafeHelper.Integer32Object deviceObj = UnsafeHelper.createInteger32Object()) {
            try {
                Object callable = getSymbol(CUDADriverFunction.CU_DEVICEGET);
                Object result = INTEROP.execute(callable, deviceObj.getAddress(), deviceOrdinal);
                checkCUReturnCode(result, "cuDeviceGet");
                return deviceObj.getValue();
            } catch (InteropException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @TruffleBoundary
    private long cuCtxCreate(int flags, int cudevice) {
        try (UnsafeHelper.PointerObject pctx = UnsafeHelper.createPointerObject()) {
            try {
                Object callable = getSymbol(CUDADriverFunction.CU_CTXCREATE);
                Object result = INTEROP.execute(callable, pctx.getAddress(), flags, cudevice);
                checkCUReturnCode(result, "cuCtxCreate");
                return pctx.getValueOfPointer();
            } catch (InteropException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @TruffleBoundary
    private long cuDevicePrimaryCtxRetain(int cudevice) {
        try (UnsafeHelper.PointerObject pctx = UnsafeHelper.createPointerObject()) {
            try {
                Object callable = getSymbol(CUDADriverFunction.CU_DEVICEPRIMARYCTXRETAIN);
                Object result = INTEROP.execute(callable, pctx.getAddress(), cudevice);
                checkCUReturnCode(result, "cuDevicePrimaryCtxRetain");
                return pctx.getValueOfPointer();
            } catch (InteropException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @TruffleBoundary
    private void cuCtxDestroy(long ctx) {
        try {
            Object callable = getSymbol(CUDADriverFunction.CU_CTXCREATE);
            Object result = INTEROP.execute(callable, ctx);
            checkCUReturnCode(result, "cuCtxDestroy");
        } catch (InteropException e) {
            throw new RuntimeException(e);
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
    private static void checkCUReturnCode(Object result, String function) {
        int returnCode;
        try {
            returnCode = INTEROP.asInt(result);
        } catch (UnsupportedMessageException e) {
            throw new RuntimeException(
                            "expected return code as Integer object in " + function + ", got " +
                                            result.getClass().getName());
        }
        if (returnCode != 0) {
            throw new CUDAException(returnCode, DriverAPIErrorMessages.getString(returnCode), function);
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
        CU_DEVICEPRIMARYCTXRETAIN("cuDevicePrimaryCtxRetain", "(pointer, sint32): sint32"),
        CU_INIT("cuInit", "(uint32): sint32"),
        CU_LAUNCHKERNEL(
                        "cuLaunchKernel",
                        "(uint64, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uint64, pointer, pointer): sint32"),
        CU_MODULELOAD("cuModuleLoad", "(pointer, string): sint32"),
        CU_MODULELOADDATA("cuModuleLoadData", "(pointer, string): sint32"),
        CU_MODULEUNLOAD("cuModuleUnload", "(uint64): sint32"),
        CU_MODULEGETFUNCTION("cuModuleGetFunction", "(pointer, uint64, string): sint32");

        final String symbolName;
        final String signature;

        CUDADriverFunction(String symbolName, String signature) {
            this.symbolName = symbolName;
            this.signature = signature;
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
