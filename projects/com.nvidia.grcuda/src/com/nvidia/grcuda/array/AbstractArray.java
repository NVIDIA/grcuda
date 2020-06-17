package com.nvidia.grcuda.array;

import com.nvidia.grcuda.ElementType;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.DefaultStream;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

/**
 * Simple wrapper around each class that represents device arrays in GrCUDA.
 * It can be used to keep track of generic arrays during execution, and monitor dependencies.
 */
@ExportLibrary(InteropLibrary.class)
public abstract class AbstractArray implements TruffleObject {

    /**
     * Reference to the underlying CUDA runtime that manages the array memory.
     */
    protected final AbstractGrCUDAExecutionContext grCUDAExecutionContext;

    /**
     * Data type of elements stored in the array.
     */
    protected final ElementType elementType;

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

    public ElementType getElementType() {
        return elementType;
    }

    protected AbstractArray(AbstractGrCUDAExecutionContext grCUDAExecutionContext, ElementType elementType) {
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.elementType = elementType;
    }

    protected AbstractArray(AbstractGrCUDAExecutionContext grCUDAExecutionContext, ElementType elementType, boolean isLastComputationArrayAccess) {
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
}
