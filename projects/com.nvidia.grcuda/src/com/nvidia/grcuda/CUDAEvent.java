package com.nvidia.grcuda;

import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.Objects;

@ExportLibrary(InteropLibrary.class)
public class CUDAEvent extends GPUPointer {

    private final long eventNumber;
    /**
     * Keep track of whether this event has been destroyed by {@link com.nvidia.grcuda.runtime.CUDARuntime#cudaEventDestroy}
     */
    private boolean isAlive = true;

    public CUDAEvent(long rawPointer, long eventNumber) {
        super(rawPointer);
        this.eventNumber = eventNumber;
    }

    public long getEventNumber() {
        return eventNumber;
    }

    public boolean isDefaultStream() { return false; }

    /**
     * Keep track of whether this event has been destroyed by {@link com.nvidia.grcuda.runtime.CUDARuntime#cudaEventDestroy}
     */
    public boolean isAlive() {
        return isAlive;
    }

    /**
     * Set the event as destroyed by the CUDA runtime;
     */
    public void setDead() {
        isAlive = false;
    }

    @Override
    public String toString() {
        return "CUDAEvent(eventNumber=" + this.eventNumber + "; address=0x" + Long.toHexString(this.getRawPointer()) + ")";
    }

    @ExportMessage
    public Object toDisplayString(boolean allowSideEffect) {
        return this.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CUDAEvent that = (CUDAEvent) o;
        return (eventNumber == that.eventNumber && this.getRawPointer() == that.getRawPointer());
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), eventNumber);
    }
}
