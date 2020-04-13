package com.nvidia.grcuda.array;

import com.nvidia.grcuda.ElementType;
import com.nvidia.grcuda.gpu.CUDARuntime;
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
    protected final CUDARuntime runtime;

    /**
     * Data type of elements stored in the array.
     */
    protected final ElementType elementType;

    /**
     * True IFF the array has been registered in {@link com.nvidia.grcuda.gpu.GrCUDAExecutionContext}.
     * Used to avoid multiple registration;
     */
    private boolean registeredInContext = false;

    public ElementType getElementType() {
        return elementType;
    }

    protected AbstractArray(CUDARuntime runtime, ElementType elementType) {
        this.runtime = runtime;
        this.elementType = elementType;
    }

    /**
     * Register the array in {@link com.nvidia.grcuda.gpu.GrCUDAExecutionContext} so that operations on this array
     * can be monitored by the runtime. Registration must be done with a separate function at the end of concrete Array classes.
     * This is done to avoid leaving the context in an inconsistent state if the concrete constructor throws an exception and fails.
     */
    protected void registerArray() {
        if (!this.registeredInContext) {
            this.runtime.getExecutionContext().registerArray(this);
            this.registeredInContext = true;
        }
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

    // TODO: equals must be smarter than checking memory address, as a MultiDimView should be considered as part of its parent
    //   The hash instead should be different. We might also not touch equals, and have another method "isPartOf"
}
