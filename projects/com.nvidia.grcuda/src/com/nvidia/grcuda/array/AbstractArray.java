package com.nvidia.grcuda.array;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.MemberSet;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.functions.DeviceArrayCopyFunction;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.DefaultStream;
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
     * Tracks whether the last operation done on the native memory underlying this array is a read/write operation
     * handled by the CPU. If so, we can avoid creating {@link com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement}
     * for array accesses that are immediately following the last one, as they are performed synchronously and there is no
     * reason to explicitly model them in the {@link ExecutionDAG};
     */
    private boolean isLastComputationArrayAccess = true;

    /** Flag set when underlying off-heap memory has been freed. */
    protected boolean arrayFreed = false;

    public Type getElementType() {
        return elementType;
    }

    protected AbstractArray(AbstractGrCUDAExecutionContext grCUDAExecutionContext, Type elementType) {
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.elementType = elementType;
    }

    protected AbstractArray(AbstractGrCUDAExecutionContext grCUDAExecutionContext, Type elementType, boolean isLastComputationArrayAccess) {
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.elementType = elementType;
        this.isLastComputationArrayAccess = isLastComputationArrayAccess;
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

    public boolean isLastComputationArrayAccess() { return isLastComputationArrayAccess; }

    public synchronized void setLastComputationArrayAccess(boolean lastComputationArrayAccess) {
        isLastComputationArrayAccess = lastComputationArrayAccess;
    }

    public abstract long getPointer();
    public abstract long getSizeBytes();
    public abstract void freeMemory();

    public boolean isMemoryFreed() {
        return arrayFreed;
    }

    /**
     * Check if this array can be accessed by the host (read/write) without having to schedule a {@link com.nvidia.grcuda.gpu.computation.ArrayAccessExecution}.
     * This is possible if the last computation on this array was also a host array access,
     * and the array is not exposed on the default stream while other GPU computations are running.
     * @return if this array can be accessed by the host without scheduling a computation
     */
    protected boolean canSkipScheduling() {
        return this.isLastComputationArrayAccess() && !(this.streamMapping.isDefaultStream() && grCUDAExecutionContext.isAnyComputationActive());
    }

    // Implementation of InteropLibrary

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
                throw ArityException.create(0, arguments.length);
            }
            freeMemory();
            return NoneValue.get();
        }
    }
}
