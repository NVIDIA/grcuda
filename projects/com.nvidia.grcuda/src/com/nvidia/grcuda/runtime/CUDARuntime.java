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
package com.nvidia.grcuda.runtime;

import com.nvidia.grcuda.CUDAEvent;
import static com.nvidia.grcuda.functions.Function.checkArgumentLength;
import static com.nvidia.grcuda.functions.Function.expectInt;
import static com.nvidia.grcuda.functions.Function.expectLong;
import static com.nvidia.grcuda.functions.Function.expectPositiveLong;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import com.nvidia.grcuda.GrCUDALogger;
import com.oracle.truffle.api.TruffleLogger;
import org.graalvm.collections.Pair;

import com.nvidia.grcuda.Binding;
import com.nvidia.grcuda.FunctionBinding;
import com.nvidia.grcuda.GPUPointer;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.Namespace;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.functions.CUDAFunction;
import com.nvidia.grcuda.runtime.UnsafeHelper.Integer32Object;
import com.nvidia.grcuda.runtime.UnsafeHelper.Integer64Object;
import com.nvidia.grcuda.runtime.computation.streamattach.StreamAttachArchitecturePolicy;
import com.nvidia.grcuda.runtime.computation.streamattach.PostPascalStreamAttachPolicy;
import com.nvidia.grcuda.runtime.computation.streamattach.PrePascalStreamAttachPolicy;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.stream.DefaultStream;
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

    private int lastDeviceSet = 0;
    private final GrCUDAContext context;
    private final NVRuntimeCompiler nvrtc;

    private final List<GPUPointer> innerCudaContexts = new ArrayList<>();
    private final int numDevices;

    public static final TruffleLogger RUNTIME_LOGGER = GrCUDALogger.getLogger(GrCUDALogger.RUNTIME_LOGGER);

    /**
     * Users can manually create streams that are not managed directly by a {@link com.nvidia.grcuda.runtime.stream.GrCUDAStreamManager}.
     * We keep track of how many of these streams have been created;
     */
    private int numUserAllocatedStreams = 0;

    public void incrementNumStreams() {
        numUserAllocatedStreams++;
    }

    public int getNumStreams() {
        return numUserAllocatedStreams;
    }

    /**
     * CUDA events are used to synchronize stream computations, and guarantee that a computation
     * starts only when all computations that depends from it are completed. Keep track of the
     * number of events created;
     */
    private long numEvents = 0;

    public void incrementNumEvents() {
        numEvents++;
    }

    public long getNumEvents() {
        return numEvents;
    }

    /**
     * Map from library-path to NFI library.
     */
    private final HashMap<String, TruffleObject> loadedLibraries = new HashMap<>();

    /**
     * Store one map between loaded functions and CUModules for every device;
     */
    private final List<HashMap<String, CUModule>> loadedModules = new ArrayList<>();

    /**
     * Map of (library-path, symbol-name) to callable.
     */
    private final HashMap<Pair<String, String>, Object> boundFunctions = new HashMap<>();

    /**
     * Depending on the available GPU, use a different policy to associate managed memory arrays to streams,
     * as specified in {@link StreamAttachArchitecturePolicy}
     */
    private final StreamAttachArchitecturePolicy streamAttachArchitecturePolicy;

    /**
     * True if the GPU architecture is Pascal or newer;
     */
    private final boolean architectureIsPascalOrNewer;

    public CUDARuntime(GrCUDAContext context, Env env) {
        this.context = context;
        try {
            TruffleObject libcudart = (TruffleObject) env.parseInternal(
                            Source.newBuilder("nfi", "load " + "lib" + CUDA_RUNTIME_LIBRARY_NAME + ".so", "cudaruntime").build()).call();
            TruffleObject libcuda = (TruffleObject) env.parseInternal(
                            Source.newBuilder("nfi", "load " + "lib" + CUDA_LIBRARY_NAME + ".so", "cuda").build()).call();
            TruffleObject libnvrtc = (TruffleObject) env.parseInternal(
                            Source.newBuilder("nfi", "load " + "lib" + NVRTC_LIBRARY_NAME + ".so", "nvrtc").build()).call();
            this.loadedLibraries.put(CUDA_RUNTIME_LIBRARY_NAME, libcudart);
            this.loadedLibraries.put(CUDA_LIBRARY_NAME, libcuda);
            this.loadedLibraries.put(NVRTC_LIBRARY_NAME, libnvrtc);

            this.numDevices = cudaGetDeviceCount();
            for (int i = 0; i < this.numDevices; i++) {
                HashMap<String, CUModule> modules = new HashMap<>();
                this.loadedModules.add(modules);
            }
        } catch (UnsatisfiedLinkError e) {
            throw new GrCUDAException(e.getMessage());
        }

        this.nvrtc = new NVRuntimeCompiler(this);
        context.addDisposable(this::shutdown);

        // Check if the GPU available in the system has Compute Capability >= 6.0 (Pascal architecture);
        this.architectureIsPascalOrNewer = cudaDeviceGetAttribute(CUDADeviceAttribute.COMPUTE_CAPABILITY_MAJOR, 0) >= 6;

        // Use pre-Pascal stream attachment policy if the CC is < 6 or if the attachment is forced by options;
        this.streamAttachArchitecturePolicy = (!this.architectureIsPascalOrNewer || context.getOptions().isForceStreamAttach()) ? new PrePascalStreamAttachPolicy() : new PostPascalStreamAttachPolicy();
    }

    // using this slow/uncached instance since all calls are non-critical
    private static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    public GrCUDAContext getContext() {
        return context;
    }

    public boolean isArchitectureIsPascalOrNewer() {
        return architectureIsPascalOrNewer;
    }

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
    public void cudaMemcpy(long destPointer, long fromPointer, long numBytesToCopy, CUDAStream stream) {
        try {
            Object callable = CUDARuntimeFunction.CUDA_MEMCPYASYNC.getSymbol(this);
            if (numBytesToCopy < 0) {
                throw new IllegalArgumentException("requested negative number of bytes to copy " + numBytesToCopy);
            }
            // cudaMemcpyKind from driver_types.h (default: direction of transfer is inferred
            // from the pointer values, uses virtual addressing)
            final long cudaMemcpyDefault = 4;
            Object result = INTEROP.execute(callable, destPointer, fromPointer, numBytesToCopy, cudaMemcpyDefault, stream.getRawPointer());
            cudaStreamSynchronize(stream);
            checkCUDAReturnCode(result, "cudaMemcpyAsync");
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
            RUNTIME_LOGGER.finest("[TRUFFLE] cudaSetDevice got called with arg " + device);
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

    @TruffleBoundary
    public CUDAStream cudaStreamCreate(int streamId) {
        try (UnsafeHelper.PointerObject streamPointer = UnsafeHelper.createPointerObject()) {
            Object callable = CUDARuntimeFunction.CUDA_STREAMCREATE.getSymbol(this);
            Object result = INTEROP.execute(callable, streamPointer.getAddress());
            checkCUDAReturnCode(result, "cudaStreamCreate");
            return new CUDAStream(streamPointer.getValueOfPointer(), streamId);
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cudaStreamSynchronize(CUDAStream stream) {
        try {
            Object callable = CUDARuntimeFunction.CUDA_STREAMSYNCHRONIZE.getSymbol(this);
            Object result = INTEROP.execute(callable, stream.getRawPointer());
            checkCUDAReturnCode(result, "cudaStreamSynchronize");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cudaStreamDestroy(CUDAStream stream) {
        try {
            Object callable = CUDARuntimeFunction.CUDA_STREAMDESTROY.getSymbol(this);
            Object result = INTEROP.execute(callable, stream.getRawPointer());
            checkCUDAReturnCode(result, "cudaStreamDestroy");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    /**
     * Limit the visibility of a managed memory array to the specified stream;
     * 
     * @param stream the stream to which we attach the array
     * @param array an array that should be assigned exclusively to a stream
     */
    @TruffleBoundary
    public void cudaStreamAttachMemAsync(CUDAStream stream, AbstractArray array) {

        final int MEM_ATTACH_SINGLE = 0x04;
        final int MEM_ATTACH_GLOBAL = 0x01;
        try {
            Object callable = CUDARuntimeFunction.CUDA_STREAMATTACHMEMASYNC.getSymbol(this);
            int flag = stream.isDefaultStream() ? MEM_ATTACH_GLOBAL : MEM_ATTACH_SINGLE;
            RUNTIME_LOGGER.finest("\t* attach array=" + System.identityHashCode(array) + " to " + stream + "; flag=" + flag);

            // Book-keeping of the stream attachment within the array;
            array.setStreamMapping(stream);

            Object result = INTEROP.execute(callable, stream.getRawPointer(), array.getFullArrayPointer(), array.getFullArraySizeBytes(), flag);
            checkCUDAReturnCode(result, "cudaStreamAttachMemAsync");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    /**
     * Synchronous version of "cudaStreamAttachMemAsync". This function doesn't exist in the CUDA
     * API, but it is useful to have;
     * 
     * @param stream the stream to which we attach the array
     * @param array an array that should be assigned exclusively to a stream
     */
    @TruffleBoundary
    public void cudaStreamAttachMem(CUDAStream stream, AbstractArray array) {
        cudaStreamAttachMemAsync(stream, array);
        cudaStreamSynchronize(stream);
    }

    @TruffleBoundary
    public void cudaMemPrefetchAsync(AbstractArray array, CUDAStream stream) {
        try {
            Object callable = CUDARuntimeFunction.CUDA_MEMPREFETCHASYNC.getSymbol(this);
            Object result = INTEROP.execute(callable, array.getFullArrayPointer(), array.getFullArraySizeBytes(), 0, stream.getRawPointer());
            checkCUDAReturnCode(result, "cudaMemPrefetchAsync");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public List<GPUPointer> getInnerCudaContexts() {
        if (this.innerCudaContexts.size() == 0) {
            assertCUDAInitialized();
        }
        return this.innerCudaContexts;
    }

    @TruffleBoundary
    public GPUPointer initializeInnerCudaContext(int deviceId) {
        int CU_CTX_SCHED_YIELD = 0x02; // Optimal multi-threaded host flag;
        return new GPUPointer(cuDevicePrimaryCtxRetain(deviceId));
    }

    /**
     * Create a new {@link CUDAEvent} and keep track of it;
     * 
     * @return a new CUDA event
     */
    @TruffleBoundary
    public CUDAEvent cudaEventCreate() {
        try (UnsafeHelper.PointerObject eventPointer = UnsafeHelper.createPointerObject()) {
            Object callable = CUDARuntimeFunction.CUDA_EVENTCREATE.getSymbol(this);
            Object result = INTEROP.execute(callable, eventPointer.getAddress());
            checkCUDAReturnCode(result, "cudaEventCreate");
            CUDAEvent event = new CUDAEvent(eventPointer.getValueOfPointer(), getNumEvents());
            incrementNumEvents();
            return event;
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    /**
     * Destroy a given CUDA event;
     * 
     * @param event a CUDA Event to destroy
     */
    @TruffleBoundary
    public void cudaEventDestroy(CUDAEvent event) {
        if (!event.isAlive()) {
            throw new RuntimeException("CUDA event=" + event + " has already been destroyed");
        }
        try {
            Object callable = CUDARuntimeFunction.CUDA_EVENTDESTROY.getSymbol(this);
            Object result = INTEROP.execute(callable, event.getRawPointer());
            checkCUDAReturnCode(result, "cudaEventDestroy");
            event.setDead();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    /**
     * Computes the elapsed time between events.
     * @param start Starting event
     * @param end Ending event
     */
    @TruffleBoundary
    public float cudaEventElapsedTime(CUDAEvent start, CUDAEvent end) {

        try(UnsafeHelper.Float32Object outPointer = UnsafeHelper.createFloat32Object()) {
            Object callable = CUDARuntimeFunction.CUDA_EVENTELAPSEDTIME.getSymbol(this);
            Object result = INTEROP.execute(callable, outPointer.getAddress(),start.getRawPointer(), end.getRawPointer());
            checkCUDAReturnCode(result, "cudaEventElapsedTime");
            float time = outPointer.getValue();
            return time;
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    /**
     * Add a given event to a stream. The event is a stream-ordered checkpoint on which we can
     * perform synchronization, or force another stream to wait for the event to occur before
     * executing any other scheduled operation queued on that stream;
     * 
     * @param event a CUDA event
     * @param stream a CUDA stream to which the event is associated
     */
    @TruffleBoundary
    public void cudaEventRecord(CUDAEvent event, CUDAStream stream) {
        if (!event.isAlive()) {
            throw new RuntimeException("CUDA event=" + event + " has already been destroyed");
        }
        try {
            Object callable = CUDARuntimeFunction.CUDA_EVENTRECORD.getSymbol(this);
            Object result = INTEROP.execute(callable, event.getRawPointer(), stream.getRawPointer());
            checkCUDAReturnCode(result, "cudaEventRecord");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    /**
     * Tell a stream to wait for a given event to occur on another stream before executing any other
     * computation;
     * 
     * @param stream a CUDA stream to which the event is associated
     * @param event a CUDA event that the stream should wait for
     */
    @TruffleBoundary
    public void cudaStreamWaitEvent(CUDAStream stream, CUDAEvent event) {
        if (!event.isAlive()) {
            throw new RuntimeException("CUDA event=" + event + " has already been destroyed");
        }
        try {
            final int FLAGS = 0x0; // Must be 0 according to CUDA documentation;
            Object callable = CUDARuntimeFunction.CUDA_STREAMWAITEVENT.getSymbol(this);
            Object result = INTEROP.execute(callable, stream.getRawPointer(), event.getRawPointer(), FLAGS);
            checkCUDAReturnCode(result, "cudaStreamWaitEvent");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    /**
     * Get function as callable from native library.
     *
     * @param binding function binding
     * @return a callable as a TruffleObject
     */
    @TruffleBoundary
    public Object getSymbol(FunctionBinding binding) throws UnknownIdentifierException {
        return getSymbol(binding.getLibraryFileName(), binding.getSymbolName(), binding.toNFISignature(), "");
    }

    /**
     * Get function as callable from native library.
     *
     * @param libraryPath path to library (.so file)
     * @param symbolName name of the function (symbol) too look up
     * @param nfiSignature NFI signature of the function
     * @return a callable as a TruffleObject
     */
    @TruffleBoundary
    public Object getSymbol(String libraryPath, String symbolName, String nfiSignature) throws UnknownIdentifierException {
        return getSymbol(libraryPath, symbolName, nfiSignature, "");
    }

    /**
     * Get function as callable from native library.
     *
     * @param libraryPath path to library (.so file)
     * @param symbolName name of the function (symbol) too look up
     * @param nfiSignature NFI signature of the function
     * @param hint additional string shown to user when symbol cannot be loaded
     * @return a callable as a TruffleObject
     */
    @TruffleBoundary
    public Object getSymbol(String libraryPath, String symbolName, String nfiSignature, String hint) throws UnknownIdentifierException {

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
                callable = INTEROP.invokeMember(symbol, "bind", nfiSignature);
            } catch (UnsatisfiedLinkError | UnsupportedMessageException | ArityException | UnsupportedTypeException e) {
                throw new GrCUDAException("unexpected behavior: " + e.getMessage());
            }
            boundFunctions.put(functionKey, callable);
        }
        return callable;
    }

    public void setLastDeviceUsed(int device) {
        this.lastDeviceSet = device;
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

    public StreamAttachArchitecturePolicy getArrayStreamArchitecturePolicy() {
        return streamAttachArchitecturePolicy;
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
        CUDA_MALLOCMANAGED("cudaMallocManaged", "(pointer, uint64, uint32): sint32") {
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
                if (cudaRuntime.context.getOptions().isEnableMultiGPU()) {
                    cudaRuntime.setLastDeviceUsed(device);
                }
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
        },
        CUDA_MEMCPYASYNC("cudaMemcpyAsync", "(pointer, pointer, uint64, sint32, pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 3);
                long destPointer = expectLong(args[0]);
                long fromPointer = expectLong(args[1]);
                long numBytesToCopy = expectPositiveLong(args[2]);
                long streamPointer = expectLong(args[3]);
                // cudaMemcpyKind from driver_types.h (default: direction of transfer is
                // inferred from the pointer values, uses virtual addressing)
                final long cudaMemcpyDefault = 4;
                callSymbol(cudaRuntime, destPointer, fromPointer, numBytesToCopy, cudaMemcpyDefault, streamPointer);
                return NoneValue.get();
            }
        },
        CUDA_STREAMCREATE("cudaStreamCreate", "(pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 0);
                try (UnsafeHelper.PointerObject streamPointer = UnsafeHelper.createPointerObject()) {
                    callSymbol(cudaRuntime, streamPointer.getAddress());
                    CUDAStream stream = new CUDAStream(streamPointer.getValueOfPointer(), cudaRuntime.getNumStreams());
                    cudaRuntime.incrementNumStreams();
                    return stream;
                }
            }
        },
        CUDA_STREAMSYNCHRONIZE("cudaStreamSynchronize", "(pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 1);
                Object pointerObj = args[0];
                long addr;
                if (pointerObj instanceof CUDAStream) {
                    addr = ((CUDAStream) pointerObj).getRawPointer();
                } else {
                    throw new GrCUDAException("expected CUDAStream object");
                }
                callSymbol(cudaRuntime, addr);
                return NoneValue.get();
            }
        },
        CUDA_STREAMDESTROY("cudaStreamDestroy", "(pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 1);
                Object pointerObj = args[0];
                long addr;
                if (pointerObj instanceof CUDAStream) {
                    addr = ((CUDAStream) pointerObj).getRawPointer();
                } else {
                    throw new GrCUDAException("expected CUDAStream object");
                }
                callSymbol(cudaRuntime, addr);
                return NoneValue.get();
            }
        },
        CUDA_STREAMATTACHMEMASYNC("cudaStreamAttachMemAsync", "(pointer, pointer, uint64, uint32): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {

                Object streamObj;
                Object arrayObj;
                final int MEM_ATTACH_SINGLE = 0x04;
                final int MEM_ATTACH_GLOBAL = 0x01;
                int flag = MEM_ATTACH_SINGLE;

                if (args.length == 1) {
                    arrayObj = args[0];
                    streamObj = DefaultStream.get();
                    flag = MEM_ATTACH_GLOBAL;
                } else if (args.length == 2) {
                    streamObj = args[0];
                    arrayObj = args[1];
                } else if (args.length == 3) {
                    streamObj = args[0];
                    arrayObj = args[1];
                    if (args[2] instanceof Integer) {
                        flag = ((Integer) args[2]);
                    } else {
                        throw new GrCUDAException("expected Integer object");
                    }
                } else {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(1, 3, args.length);
                }

                // Extract pointers;
                long streamAddr;
                long arrayAddr;
                streamAddr = extractStreamPointer(streamObj);
                arrayAddr = extractArrayPointer(arrayObj);

                // If using the default stream (0 address) use the "cudaMemAttachGlobal" flag;
                if (streamAddr == 0) {
                    flag = MEM_ATTACH_GLOBAL;
                }

                // Track the association between the stream and the array, if possible;
                if (streamObj instanceof CUDAStream) {
                    if (arrayObj instanceof AbstractArray) {
                        ((AbstractArray) arrayObj).setStreamMapping((CUDAStream) streamObj);
                    }
                }

                // Always set "size" to 0 to cover the entire array;
                callSymbol(cudaRuntime, streamAddr, arrayAddr, 0, flag);
                return NoneValue.get();
            }
        },
        CUDA_MEMPREFETCHASYNC("cudaMemPrefetchAsync", "(pointer, uint64, sint32, pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {

                Object streamObj;
                Object arrayObj;
                long size;
                int destinationDevice;

                if (args.length == 3) {
                    arrayObj = args[0];
                    streamObj = DefaultStream.get();
                } else if (args.length == 4) {
                    arrayObj = args[0];
                    streamObj = args[3];
                } else {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(3, 4, args.length);
                }

                if (args[1] instanceof Long) {
                    size = ((Long) args[1]);
                } else {
                    throw new GrCUDAException("expected Long object for array size");
                }

                if (args[2] instanceof Integer) {
                    destinationDevice = ((Integer) args[2]);
                } else {
                    throw new GrCUDAException("expected Integer object for destination device");
                }

                // Extract pointers;
                long streamAddr;
                long arrayAddr;
                streamAddr = extractStreamPointer(streamObj);
                arrayAddr = extractArrayPointer(arrayObj);

                // Always set "size" to 0 to cover the entire array;
                callSymbol(cudaRuntime, arrayAddr, size, destinationDevice, streamAddr);
                return NoneValue.get();
            }
        },
        CUDA_EVENTCREATE("cudaEventCreate", "(pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 0);
                try (UnsafeHelper.PointerObject eventPointer = UnsafeHelper.createPointerObject()) {
                    callSymbol(cudaRuntime, eventPointer.getAddress());
                    CUDAEvent event = new CUDAEvent(eventPointer.getValueOfPointer(), cudaRuntime.getNumEvents());
                    cudaRuntime.incrementNumEvents();
                    return event;
                }
            }
        },
        CUDA_EVENTDESTROY("cudaEventDestroy", "(pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 1);
                Object pointerObj = args[0];
                long addr;
                if (pointerObj instanceof CUDAEvent) {
                    addr = ((CUDAEvent) pointerObj).getRawPointer();
                } else {
                    throw new GrCUDAException("expected CUDAEvent object");
                }
                callSymbol(cudaRuntime, addr);
                return NoneValue.get();
            }
        },
        CUDA_EVENTELAPSEDTIME("cudaEventElapsedTime", "(pointer, pointer, pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 3);

                Object pointerStartEvent = args[1];
                Object pointerEndEvent = args[2];
                long addrStart;
                long addrEnd;

                if (pointerStartEvent instanceof CUDAEvent) {
                    addrStart = ((CUDAEvent) pointerStartEvent).getRawPointer();
                } else {
                    throw new GrCUDAException("expected CUDAEvent object");
                }
                if (pointerEndEvent instanceof CUDAEvent) {
                    addrEnd = ((CUDAEvent) pointerEndEvent).getRawPointer();
                } else {
                    throw new GrCUDAException("expected CUDAEvent object");
                }
                try (UnsafeHelper.Float32Object floatPointer = UnsafeHelper.createFloat32Object()) {
                    callSymbol(cudaRuntime, floatPointer.getAddress(), addrStart, addrEnd );
                    return NoneValue.get();
                }
            }
        },
        CUDA_EVENTRECORD("cudaEventRecord", "(pointer, pointer): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 2);
                Object eventObj = args[0];
                Object streamObj = args[1];
                long eventAddr, streamAddr;
                if (eventObj instanceof CUDAEvent) {
                    eventAddr = ((CUDAEvent) eventObj).getRawPointer();
                } else {
                    throw new GrCUDAException("expected CUDAEvent object");
                }
                if (streamObj instanceof CUDAStream) {
                    streamAddr = ((CUDAStream) streamObj).getRawPointer();
                } else {
                    throw new GrCUDAException("expected CUDAStream object");
                }
                callSymbol(cudaRuntime, eventAddr, streamAddr);
                return NoneValue.get();
            }
        },
        CUDA_STREAMWAITEVENT("cudaStreamWaitEvent", "(pointer, pointer, uint32): sint32") {
            @Override
            @TruffleBoundary
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, UnsupportedTypeException, InteropException {
                checkArgumentLength(args, 2);
                Object streamObj = args[0];
                Object eventObj = args[1];
                long streamAddr, eventAddr;
                final int FLAGS = 0x0; // Flags must be zero according to CUDA documentation;

                if (streamObj instanceof CUDAStream) {
                    streamAddr = ((CUDAStream) streamObj).getRawPointer();
                } else {
                    throw new GrCUDAException("expected CUDAStream object");
                }
                if (eventObj instanceof CUDAEvent) {
                    eventAddr = ((CUDAEvent) eventObj).getRawPointer();
                } else {
                    throw new GrCUDAException("expected CUDAEvent object");
                }
                callSymbol(cudaRuntime, streamAddr, eventAddr, FLAGS);
                return NoneValue.get();
            }
        },
        CUDA_PROFILERSTART("cudaProfilerStart", "(): sint32") {
            @Override
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, InteropException, UnsupportedMessageException {
                checkArgumentLength(args, 0);
                callSymbol(cudaRuntime);
                return NoneValue.get();
            }
        },
        CUDA_PROFILERSTOP("cudaProfilerStop", "(): sint32") {
            @Override
            public Object call(CUDARuntime cudaRuntime, Object[] args) throws ArityException, InteropException, UnsupportedMessageException {
                checkArgumentLength(args, 0);
                callSymbol(cudaRuntime);
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

        long extractArrayPointer(Object array) {
            if (array instanceof GPUPointer) {
                return ((GPUPointer) array).getRawPointer();
            } else if (array instanceof LittleEndianNativeArrayView) {
                return ((LittleEndianNativeArrayView) array).getStartAddress();
            } else if (array instanceof AbstractArray) {
                return ((AbstractArray) array).getFullArrayPointer();
            } else {
                throw new GrCUDAException("expected GPUPointer or LittleEndianNativeArrayView or DeviceArray");
            }
        }

        long extractStreamPointer(Object stream) {
            if (stream instanceof CUDAStream) {
                return ((CUDAStream) stream).getRawPointer();
            } else {
                throw new GrCUDAException("expected CUDAStream object");
            }
        }
    }

// @TruffleBoundary
// public Kernel loadKernel(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String cubinFile,
// String kernelName, String signature) {
// CUModule module = loadedModules.get(cubinFile);
// try {
// if (module == null) {
// module = cuModuleLoad(cubinFile);
// }
// long kernelFunction = cuModuleGetFunction(module, kernelName);
// return new Kernel(grCUDAExecutionContext, kernelName, module, kernelFunction, signature);
// } catch (Exception e) {
// if ((module != null) && (module.getRefCount() == 1)) {
// cuModuleUnload(module);
// }
// throw e;

    @TruffleBoundary
    public Kernel loadKernel(AbstractGrCUDAExecutionContext grCUDAExecutionContext, Binding binding) {
        if (this.context.getOptions().isEnableMultiGPU()) {
            return this.loadKernelMultiGPU(grCUDAExecutionContext, binding.getLibraryFileName(), binding.getName(), binding.getSymbolName(), binding.getNIDLParameterSignature());
        } else {
            return this.loadKernelSingleGPU(grCUDAExecutionContext, binding.getLibraryFileName(), binding.getName(), binding.getSymbolName(), binding.getNIDLParameterSignature());
        }
    }

    @TruffleBoundary
    public Kernel loadKernelSingleGPU(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String cubinFile, String kernelName, String symbolName, String signature) {
        // Load module from GPU 0;
        CUModule module = loadedModules.get(0).get(cubinFile);
        if (module == null) {
            // load module as it is not yet loaded
            module = cuModuleLoad(cubinFile);
            loadedModules.get(0).put(cubinFile, module);
        }
        long kernelFunctionHandle = cuModuleGetFunction(module, symbolName);
        return new Kernel(grCUDAExecutionContext, kernelName, symbolName, List.of(kernelFunctionHandle), signature, List.of(module));
    }

    @TruffleBoundary
    public Kernel loadKernelMultiGPU(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String cubinFile, String kernelName, String symbolName, String signature) {

        ArrayList<Long> kernelFunctionHandles = new ArrayList<>();
        ArrayList<CUModule> modules = new ArrayList<>();
        int currentDevice = cudaGetDevice();

        for (int i = 0; i < numDevices; i++) {
            CUModule module = loadedModules.get(i).get(cubinFile);
            cudaSetDevice(i);
            if (module == null) {
                module = cuModuleLoad(cubinFile);
                loadedModules.get(i).put(cubinFile, module);
            }
            modules.add(module);

            long kernelFunctionHandle = cuModuleGetFunction(module, symbolName);
            kernelFunctionHandles.add(kernelFunctionHandle);
        }
        cudaSetDevice(currentDevice);

        return new Kernel(grCUDAExecutionContext, kernelName, symbolName, kernelFunctionHandles, signature, modules);
    }

    @TruffleBoundary
    public Kernel buildKernel(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String code, String kernelName, String signature) {
        RUNTIME_LOGGER.finest("buildKernel device:" + cudaGetDevice());
        String moduleName = "truffle" + context.getNextModuleId();
        PTXKernel ptx = nvrtc.compileKernel(code, kernelName, moduleName, "--std=c++14");
        if (this.context.getOptions().isEnableMultiGPU()) {
            return this.buildKernelMultiGPU(grCUDAExecutionContext, kernelName, signature, moduleName, ptx);
        } else {
            return this.buildKernelSingleGPU(grCUDAExecutionContext, kernelName, signature, moduleName, ptx);
        }
    }

    public Kernel buildKernelSingleGPU(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String kernelName, String signature, String moduleName, PTXKernel ptx) {
        CUModule module = cuModuleLoadData(ptx.getPtxSource(), moduleName);
        loadedModules.get(0).put(moduleName, module);
        long kernelFunctionHandle = cuModuleGetFunction(module, ptx.getLoweredKernelName());
        return new Kernel(grCUDAExecutionContext, kernelName, ptx.getLoweredKernelName(), List.of(kernelFunctionHandle),
                signature, List.of(module), ptx.getPtxSource());
    }

    public Kernel buildKernelMultiGPU(AbstractGrCUDAExecutionContext grCUDAExecutionContext, String kernelName, String signature, String moduleName, PTXKernel ptx) {
        ArrayList<Long> kernelFunctionHandles = new ArrayList<>();
        ArrayList<CUModule> modules = new ArrayList<>();
        int currentDevice = cudaGetDevice();

        for (int i = 0; i < numDevices; i++) {
            cudaSetDevice(i);
            CUModule module = cuModuleLoadData(ptx.getPtxSource(), moduleName);
            long kernelFunctionHandle = cuModuleGetFunction(module, ptx.getLoweredKernelName());
            kernelFunctionHandles.add(kernelFunctionHandle);
            modules.add(module);
            loadedModules.get(i).put(moduleName, module);
        }
        cudaSetDevice(currentDevice);
        return new Kernel(grCUDAExecutionContext, kernelName, ptx.getLoweredKernelName(), kernelFunctionHandles,
                signature, modules, ptx.getPtxSource());
    }

    @TruffleBoundary
    public CUModule cuModuleLoad(String cubinName) {
        assertCUDAInitialized();
        int currDevice = this.context.getOptions().isEnableMultiGPU() ? cudaGetDevice() : 0;
        if (this.loadedModules.get(currDevice).containsKey(cubinName)) {
            throw new GrCUDAException("A module for " + cubinName + " was already loaded.");
        }
        try (UnsafeHelper.Integer64Object modulePtr = UnsafeHelper.createInteger64Object()) {
            Object callable = CUDADriverFunction.CU_MODULELOAD.getSymbol(this);
            Object result = INTEROP.execute(callable, modulePtr.getAddress(), cubinName);
            checkCUReturnCode(result, "cuModuleLoad");
            return new CUModule(cubinName, modulePtr.getValue());
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public CUModule cuModuleLoadData(String ptx, String moduleName) {
        assertCUDAInitialized();
        int currDevice = this.context.getOptions().isEnableMultiGPU() ? cudaGetDevice() : 0;
        if (this.loadedModules.get(currDevice).containsKey(moduleName)) {
            throw new GrCUDAException("A module for " + moduleName + " was already loaded.");
        }
        try (UnsafeHelper.Integer64Object modulePtr = UnsafeHelper.createInteger64Object()) {
            Object callable = CUDADriverFunction.CU_MODULELOADDATA.getSymbol(this);
            Object result = INTEROP.execute(callable,
                            modulePtr.getAddress(), ptx);
            checkCUReturnCode(result, "cuModuleLoadData");
            return new CUModule(moduleName, modulePtr.getValue());
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cuModuleUnload(CUModule module) {
        try {
            Object callable = CUDADriverFunction.CU_MODULEUNLOAD.getSymbol(this);
            Object result = INTEROP.execute(callable, module.modulePointer);
            checkCUReturnCode(result, "cuModuleUnload");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    /**
     * Get function handle to kernel in module.
     *
     * @param kernelModule CUmodule containing the kernel function
     * @param kernelName
     * @return native CUfunction function handle
     */
    public long cuModuleGetFunction(CUModule kernelModule, String kernelName) {
        try (UnsafeHelper.Integer64Object functionPtr = UnsafeHelper.createInteger64Object()) {
            Object callable = CUDADriverFunction.CU_MODULEGETFUNCTION.getSymbol(this);
            Object result = INTEROP.execute(callable,
                            functionPtr.getAddress(), kernelModule.getModulePointer(), kernelName);
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
        this.cuLaunchKernel(kernel, config, args, config.getStream());
    }

    @TruffleBoundary
    public void cuLaunchKernel(Kernel kernel, KernelConfig config, KernelArguments args, CUDAStream stream) {

        // FIXME: as multigpu is not ~properly~ supported in this branch
        // using multiple gpu requires the stream associated to the device to be
        // set manually.
        long kernelFunctionHandle;
        if (this.context.getOptions().isEnableMultiGPU()) {
            stream.setDeviceId(this.lastDeviceSet);
            cudaSetDevice(stream.getStreamDeviceId());
            assert stream.getStreamDeviceId() == cudaGetDevice();
            kernelFunctionHandle = kernel.getKernelFunctionHandle(stream.getStreamDeviceId());
        } else {
            kernelFunctionHandle = kernel.getKernelFunctionHandle(0);
        }

        try {

            Object callable = CUDADriverFunction.CU_LAUNCHKERNEL.getSymbol(this);
            Dim3 gridSize = config.getGridSize();
            Dim3 blockSize = config.getBlockSize();
            Object result = INTEROP.execute(callable,
                    kernelFunctionHandle,
                            gridSize.getX(),
                            gridSize.getY(),
                            gridSize.getZ(),
                            blockSize.getX(),
                            blockSize.getY(),
                            blockSize.getZ(),
                            config.getDynamicSharedMemoryBytes(),
                            stream.getRawPointer(),
                            args.getPointer(),              // pointer to kernel arguments array
                            0                               // extra args
            );
            checkCUReturnCode(result, "cuLaunchKernel");
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
    private long cuCtxGetCurrent() {
        try (UnsafeHelper.PointerObject ctxPointer = UnsafeHelper.createPointerObject()) {
            Object callable = CUDADriverFunction.CU_CTXGETCURRENT.getSymbol(this);
            Object result = INTEROP.execute(callable, ctxPointer.getAddress());
            checkCUDAReturnCode(result, "cuCtxGetCurrent");
            return ctxPointer.getValueOfPointer();
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    public void cuCtxSetCurrent(GPUPointer ctxPointer) {
        try {
            Object callable = CUDADriverFunction.CU_CTXSETCURRENT.getSymbol(this);
            Object result = INTEROP.execute(callable, ctxPointer.getRawPointer());
            checkCUDAReturnCode(result, "cuCtxSetCurrent");
        } catch (InteropException e) {
            throw new GrCUDAException(e);
        }
    }

    @TruffleBoundary
    private void assertCUDAInitialized() {
        if (!context.isCUDAInitialized()) {
            int currentDevice = cudaGetDevice();
            if (this.context.getOptions().isEnableMultiGPU()) {
                int numDevices = cudaGetDeviceCount();
                for (int i = 0; i < numDevices; i++) {
                    cudaSetDevice(i);
                    cuInit();
                    // A simple way to create the device context in the driver is to call any CUDA
                    // API function;
                    cudaDeviceSynchronize();
                    this.innerCudaContexts.add(initializeInnerCudaContext(i));
                }
                cudaSetDevice(currentDevice);
            } else {
                cuInit();
                // A simple way to create the device context in the driver is to call any CUDA API
                // function;
                cudaDeviceSynchronize();
                // Initialize only device 0;
                this.innerCudaContexts.add(initializeInnerCudaContext(0));
            }
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
            RUNTIME_LOGGER.severe("ERROR CODE=" + returnCode);
            throw new GrCUDAException(returnCode, DriverAPIErrorMessages.getString(returnCode), function);

        }
    }

    private void shutdown() {
        // unload all modules
        for (int i = 0; i < numDevices; i++) {
            for (CUModule module : loadedModules.get(i).values()) {
                try {
                    module.close();
                } catch (Exception e) {
                    /* ignore exception */
                }
            }
            loadedModules.get(i).clear();
        }
    }

    public enum CUDADriverFunction {
        CU_CTXCREATE("cuCtxCreate", "(pointer, uint32, sint32) :sint32"),
        CU_CTXDESTROY("cuCtxDestroy", "(pointer): sint32"),
        CU_CTXGETCURRENT("cuCtxGetCurrent", "(pointer) :sint32"),
        CU_CTXSETCURRENT("cuCtxSetCurrent", "(pointer) :sint32"),
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

    final class CUModule implements AutoCloseable {
        final String cubinFile;
        /** Pointer to the native CUmodule object. */
        final long modulePointer;
        boolean closed = false;

        CUModule(String cubinFile, long modulePointer) {
            this.cubinFile = cubinFile;
            this.modulePointer = modulePointer;
            this.closed = false;
        }

        public long getModulePointer() {
            if (closed) {
                CompilerDirectives.transferToInterpreter();
                throw new GrCUDAException(String.format("cannot get module pointer, module (%016x) already closed", modulePointer));
            }
            return modulePointer;
        }

        public boolean isClosed() {
            return closed;
        }

        @Override
        public boolean equals(Object other) {
            if (other instanceof CUModule) {
                CUModule otherModule = (CUModule) other;
                return otherModule.cubinFile.equals(cubinFile) && otherModule.closed == closed;
            } else {
                return false;
            }
        }

        @Override
        public int hashCode() {
            return cubinFile.hashCode();
        }

        @Override
        public void close() {
            if (!closed) {
                cuModuleUnload(this);
                closed = true;
            }
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
