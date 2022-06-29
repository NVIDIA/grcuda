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
package com.nvidia.grcuda.runtime.array;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.MemberSet;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.LittleEndianNativeArrayView;
import com.nvidia.grcuda.runtime.computation.arraycomputation.ArrayAccessExecution;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.stream.DefaultStream;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

import java.util.HashSet;
import java.util.Set;

/**
 * Simple wrapper around each class that represents device arrays in GrCUDA.
 * It can be used to keep track of generic arrays during execution, and monitor dependencies.
 */
@ExportLibrary(InteropLibrary.class)
public abstract class AbstractArray implements TruffleObject {

    protected static final String POINTER = "pointer";
    protected static final String COPY_FROM = "copyFrom";
    protected static final String COPY_TO = "copyTo";
    protected static final String FREE = "free";
    protected static final String IS_MEMORY_FREED = "isMemoryFreed";
    protected static final String ACCESSED_FREED_MEMORY_MESSAGE = "memory of array freed";

    protected static final MemberSet PUBLIC_MEMBERS = new MemberSet(COPY_FROM, COPY_TO, FREE, IS_MEMORY_FREED);
    protected static final MemberSet MEMBERS = new MemberSet(POINTER, COPY_FROM, COPY_TO, FREE, IS_MEMORY_FREED);

    /**
     * Reference to the underlying CUDA runtime that manages the array memory.
     */
    protected final AbstractGrCUDAExecutionContext grCUDAExecutionContext;

    /**
     * Data type of elements stored in the array.
     */
    protected final Type elementType;

    /**
     * True IFF the array has been registered in {@link AbstractGrCUDAExecutionContext}.
     * Used to avoid multiple registration;
     */
    private boolean registeredInContext = false;

    /**
     * Keep track of whether this array is attached to a specific stream that limits its visibility.
     * By default, every array is attached to the {@link DefaultStream};
     */
    protected CUDAStream streamMapping = DefaultStream.get();

    /**
     * Function used to compute if we can skip the scheduling of a computational element for a given array read;
     */
    private final SkipSchedulingInterface skipScheduleRead;
    /**
     * Function used to compute if we can skip the scheduling of a computational element for a given array write;
     */
    private final SkipSchedulingInterface skipScheduleWrite;

    /** Flag set when underlying off-heap memory has been freed. */
    protected boolean arrayFreed = false;

    /**
     * List of devices where the array is currently UP-TO-DATE, i.e. it can be accessed without requiring any memory transfer.
     * On pre-Pascal GPUs, arrays are allocated on the currently active GPU. On devices since Pascal, arrays are allocated on the CPU.
     * We identify devices using integers. CPU is -1 ({@link com.nvidia.grcuda.runtime.CPUDevice#CPU_DEVICE_ID}, GPUs start from 0;
     */
    protected final Set<Integer> arrayUpToDateLocations = new HashSet<>();

    public Type getElementType() {
        return elementType;
    }

    protected AbstractArray(AbstractGrCUDAExecutionContext grCUDAExecutionContext, Type elementType) {
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.elementType = elementType;

        // Specify how we can identify if we can skip the scheduling of array accesses;
        if (this.grCUDAExecutionContext.isArchitecturePascalOrNewer()) {
            this.skipScheduleRead = this::isArrayUpdatedOnCPU;
            this.skipScheduleWrite = this::isArrayUpdatedOnlyOnCPU;
        } else {
            // On pre-Pascal devices, we cannot access GPU memory while some other computation is active on the default stream;
            this.skipScheduleRead = () -> isArrayUpdatedOnCPU() && !(streamMapping.isDefaultStream() && grCUDAExecutionContext.isAnyComputationActive());
            this.skipScheduleWrite = () -> isArrayUpdatedOnlyOnCPU() && !(streamMapping.isDefaultStream() && grCUDAExecutionContext.isAnyComputationActive());
        }
        // Initialize the location of an abstract array.
        // On pre-Pascal devices, the default location is the current GPU. Since Pascal, it is the CPU.
        if (this.grCUDAExecutionContext.isArchitecturePascalOrNewer()) {
            this.addArrayUpToDateLocations(CPUDevice.CPU_DEVICE_ID);
        } else {
            this.addArrayUpToDateLocations(grCUDAExecutionContext.getCurrentGPU());
        }
    }

    protected AbstractArray(AbstractArray otherArray) {
        this.grCUDAExecutionContext = otherArray.grCUDAExecutionContext;
        this.elementType = otherArray.elementType;
        this.skipScheduleRead = otherArray.skipScheduleRead;
        this.skipScheduleWrite = otherArray.skipScheduleWrite;
        // Initialize the location of an abstract array, copying the ones specified in the input;
        this.arrayUpToDateLocations.addAll(otherArray.getArrayUpToDateLocations());
        this.arrayFreed = otherArray.arrayFreed;
        this.streamMapping = otherArray.streamMapping;
        // Registration must be done afterwards;
        this.registeredInContext = false;
    }

    /**
     * Register the array in {@link AbstractGrCUDAExecutionContext} so that operations on this array
     * can be monitored by the runtime. Registration must be done with a separate function at the end of concrete Array classes.
     * This is done to avoid leaving the context in an inconsistent state if the concrete constructor throws an exception and fails.
     */
    protected void registerArray() {
        if (!this.registeredInContext) {
            this.grCUDAExecutionContext.registerArray(this);
            this.registeredInContext = true;
        }
    }

    public AbstractGrCUDAExecutionContext getGrCUDAExecutionContext() {
        return grCUDAExecutionContext;
    }

    public CUDAStream getStreamMapping() {
        return streamMapping;
    }

    public void setStreamMapping(CUDAStream streamMapping) {
        this.streamMapping = streamMapping;
    }

    /**
     * Tracks whether the array is up-to-date on CPU.
     * This happens if the last operation done on the native memory underlying this array is a read/write operation
     * handled by the CPU. If so, we can avoid creating {@link com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement}
     * for array reads that are immediately following the last one, as they are performed synchronously and there is no
     * reason to explicitly model them in the {@link ExecutionDAG};
     */
    // FIXME (check if fixed already): Possible error: Array A is up-to-date on CPU and GPU0. There's an ongoing kernel on GPU0 that uses A read-only.
    //  If we write A on the CPU, is the scheduling skipped? That's an error.
    //  In the case of a read, no problem (a kernel that modifies the data would take exclusive ownership),
    //  while in the case of a write we need to check that arrayUpToDateLocations == CPU
    public boolean isArrayUpdatedOnCPU() {
        return this.arrayUpToDateLocations.contains(CPUDevice.CPU_DEVICE_ID);
    }

    /**
     * Tracks whether the array is up-to-date only on CPU, and not on other devices.
     * This happens if the last operation done on the native memory underlying this array is a read/write operation
     * handled by the CPU. If so, we can avoid creating {@link com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement}
     * for array accesses that are immediately following the last one, as they are performed synchronously and there is no
     * reason to explicitly model them in the {@link ExecutionDAG}.
     * To perform a write on the CPU, we need the array to be updated exclusively on the CPU;
     */
    public boolean isArrayUpdatedOnlyOnCPU() {
        return this.arrayUpToDateLocations.size() == 1 && this.arrayUpToDateLocations.contains(CPUDevice.CPU_DEVICE_ID);
    }

    public Set<Integer> getArrayUpToDateLocations() {
        return this.arrayUpToDateLocations;
    }

    /**
     * Reset the list of devices where the array is currently up-to-date,
     * and specify a new device where the array is up-to-date.
     * Used when the array is modified by some device: there should never be a situation where
     * the array is not up-to-date on at least one device;
     */
    public void resetArrayUpToDateLocations(int deviceId) {
        this.arrayUpToDateLocations.clear();
        this.arrayUpToDateLocations.add(deviceId);
    }

    public void addArrayUpToDateLocations(int deviceId) {
        this.arrayUpToDateLocations.add(deviceId);
    }

    /**
     * True if this array is up-to-date for the input device;
     * @param deviceId a device for which we want to check if this array is up-to-date;
     * @return if this array is up-to-date with respect to the input device
     */
    public boolean isArrayUpdatedInLocation(int deviceId) {
        return this.arrayUpToDateLocations.contains(deviceId);
    }

    public abstract long getPointer();
    public abstract long getSizeBytes();
    public abstract void freeMemory();

    /**
     * Access the underlying native memory of the array, as if it were a linear 1D array.
     * It can be used to copy chunks of the array without having to perform repeated checks,
     * and for the low-level implementation of array accesses
     * @param index index used to access the array
     * @param elementTypeProfile profiling of the element type, to speed up the native view access
     * @return element of the array
     */
    public abstract Object readNativeView(long index, @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile);

    /**
     * Static method to read the native view of an array. It can be used to implement the innermost access in {@link AbstractArray#readNativeView};
     * @param nativeView native array representation of the array
     * @param index index used to access the array
     * @param elementType type of the array, required to know the size of each element
     * @param elementTypeProfile profiling of the element type, to speed up the native view access
     * @return element of the array
     */
    protected static Object readArrayElementNative(LittleEndianNativeArrayView nativeView, long index, Type elementType,
                                                   @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) {
        switch (elementTypeProfile.profile(elementType)) {
            case CHAR:
                return nativeView.getByte(index);
            case SINT16:
                return nativeView.getShort(index);
            case SINT32:
                return nativeView.getInt(index);
            case SINT64:
                return nativeView.getLong(index);
            case FLOAT:
                return nativeView.getFloat(index);
            case DOUBLE:
                return nativeView.getDouble(index);
        }
        return null;
    }

    /**
     * Access the underlying native memory of the array, as if it were a linear 1D array.
     * It can be used to copy chunks of the array without having to perform repeated checks,
     * and for the low-level implementation of array accesses
     * @param index index used to access the array
     * @param value value to write in the array
     * @param valueLibrary interop access of the value, required to understand its type
     * @param elementTypeProfile profiling of the element type, to speed up the native view access
     * @throws UnsupportedTypeException if writing the wrong type in the array
     */
    public abstract void writeNativeView(long index, Object value, @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
                                         @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws UnsupportedTypeException;

    /**
     * Static method to write the native view of an array. It can be used to implement the innermost access in {@link AbstractArray#writeNativeView};
     * @param nativeView native array representation of the array
     * @param index index used to access the array
     * @param value value to write in the array
     * @param elementType type of the array, required to know the size of each element
     * @param valueLibrary interop access of the value, required to understand its type
     * @param elementTypeProfile profiling of the element type, to speed up the native view access
     * @throws UnsupportedTypeException if writing the wrong type in the array
     */
    public static void writeArrayElementNative(LittleEndianNativeArrayView nativeView, long index, Object value, Type elementType,
                                               @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
                                               @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws UnsupportedTypeException {
        try {
            switch (elementTypeProfile.profile(elementType)) {
                case CHAR:
                    nativeView.setByte(index, valueLibrary.asByte(value));
                    break;
                case SINT16:
                    nativeView.setShort(index, valueLibrary.asShort(value));
                    break;
                case SINT32:
                    nativeView.setInt(index, valueLibrary.asInt(value));
                    break;
                case SINT64:
                    nativeView.setLong(index, valueLibrary.asLong(value));
                    break;
                case FLOAT:
                    // going via "double" to allow floats to be initialized with doubles
                    nativeView.setFloat(index, (float) valueLibrary.asDouble(value));
                    break;
                case DOUBLE:
                    nativeView.setDouble(index, valueLibrary.asDouble(value));
                    break;
            }
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{value}, "value cannot be coerced to " + elementType);
        }
    }

    public boolean isMemoryFreed() {
        return arrayFreed;
    }

    /**
     * Check if this array can be accessed by the host for a read without having to schedule a {@link ArrayAccessExecution}.
     * This is possible if the array is up-to-date on the CPU,
     * and the array is not exposed on the default stream while other GPU computations are running (on pre-Pascal devices).
     * @return if this array can be accessed by the host without scheduling a computation
     */
    public boolean canSkipSchedulingRead() {
        return this.skipScheduleRead.canSkipScheduling();
    }

    /**
     * Check if this array can be accessed by the host for a write without having to schedule a {@link ArrayAccessExecution}.
     * This is possible if the array is assumed up-to-date only on the CPU,
     * and the array is not exposed on the default stream while other GPU computations are running (on pre-Pascal devices).
     * @return if this array can be accessed by the host without scheduling a computation
     */
    public boolean canSkipSchedulingWrite() {
        return this.skipScheduleWrite.canSkipScheduling();
    }

    protected interface SkipSchedulingInterface {
        boolean canSkipScheduling();
    }

    /**
     * When working with array views, it is common to require both the pointer to the view, and the pointer
     * to the whole array. This method always returns the pointer to the whole array,
     * and should be used when dealing with low-level CUDA APIs that cannot handle just part of the array.
     * By default, i.e. if the array is not a view of a larger array,
     * this function is identical to {@link AbstractArray#getPointer()}
     * @return the pointer to the whole array
     */
    public long getFullArrayPointer() {
        return this.getPointer();
    }

    /**
     * When working with array views, it is common to require both the size of the view, and the size of
     * the whole array. This method always returns the size (in bytes) of the whole array,
     * and should be used when dealing with low-level CUDA APIs that cannot handle just part of the array.
     * By default, i.e. if the array is not a view of a larger array,
     * this function is identical to {@link AbstractArray#getSizeBytes()}
     * @return the size, in bytes, of the whole array
     */
    public long getFullArraySizeBytes() {
        return this.getSizeBytes();
    }

    /**
     * By default, we assume that arrays are stored in row-major format ("C" format).
     * This holds true for {@link DeviceArray}s, which are 1D arrays where the storage order does not matter;
     * @return if the array was stored in column-major format (i.e. "Fortran" or "F")
     */
    public boolean isColumnMajorFormat() {
        return false;
    }

    // Implementation of InteropLibrary

    @ExportMessage
    boolean isPointer() { return true; }

    @ExportMessage
    long asPointer() { return this.getPointer(); }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasArrayElements() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return true;
    }

    @ExportMessage
    Object readArrayElement(long index) throws UnsupportedMessageException, InvalidArrayIndexException {
        return null;
    }

    @ExportMessage
    boolean isArrayElementReadable(long index) {
        return false;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    Object getMembers(boolean includeInternal) {
        return includeInternal ? MEMBERS : PUBLIC_MEMBERS;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String memberName,
                             @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        String name = memberProfile.profile(memberName);
        return POINTER.equals(name) || COPY_FROM.equals(name) || COPY_TO.equals(name) || FREE.equals(name) || IS_MEMORY_FREED.equals(name);
    }

    @ExportMessage
    Object readMember(String memberName,
                      @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) throws UnknownIdentifierException {
        if (!isMemberReadable(memberName, memberProfile)) {
            CompilerDirectives.transferToInterpreter();
            throw UnknownIdentifierException.create(memberName);
        }
        if (POINTER.equals(memberName)) {
            return getPointer();
        }
        if (COPY_FROM.equals(memberName)) {
            return new DeviceArrayCopyFunction(this, DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
        }
        if (COPY_TO.equals(memberName)) {
            return new DeviceArrayCopyFunction(this, DeviceArrayCopyFunction.CopyDirection.TO_POINTER);
        }
        if (FREE.equals(memberName)) {
            return new DeviceArray.DeviceArrayFreeFunction();
        }
        if (IS_MEMORY_FREED.equals(memberName)) {
            return isMemoryFreed();
        }
        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(memberName);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberInvocable(String memberName) {
        return COPY_FROM.equals(memberName) || COPY_TO.equals(memberName) || FREE.equals(memberName);
    }

    @ExportMessage
    Object invokeMember(String memberName,
                        Object[] arguments,
                        @CachedLibrary("this") InteropLibrary interopRead,
                        @CachedLibrary(limit = "1") InteropLibrary interopExecute)
            throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        return interopExecute.execute(interopRead.readMember(this, memberName), arguments);
    }

    /**
     * Retrieve the total number of elements in the array,
     * or the size of the current dimension for matrices and tensors
     *
     * @return the total number of elements in the array
     */
    @ExportMessage
    public abstract long getArraySize();

    // TODO: equals must be smarter than checking memory address, as a MultiDimView should be considered as part of its parent,
    //   similarly to what "isLastComputationArrayAccess" is doing.
    //   The hash instead should be different. We might also not touch equals, and have another method "isPartOf"

    @ExportLibrary(InteropLibrary.class)
    final class DeviceArrayFreeFunction implements TruffleObject {
        @ExportMessage
        @SuppressWarnings("static-method")
        boolean isExecutable() {
            return true;
        }

        @ExportMessage
        Object execute(Object[] arguments) throws ArityException {
            if (arguments.length != 0) {
                CompilerDirectives.transferToInterpreter();
                throw ArityException.create(0, 0, arguments.length);
            }
            freeMemory();
            return NoneValue.get();
        }
    }
}
