package com.nvidia.grcuda;

import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.Objects;

@ExportLibrary(InteropLibrary.class)
public class CUDAEvent extends GPUPointer {

    private final long eventNumber;

    public CUDAEvent(long rawPointer, long eventNumber) {
        super(rawPointer);
        this.eventNumber = eventNumber;
    }

    public long getEventNumber() {
        return eventNumber;
    }

    public boolean isDefaultStream() { return false; }

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
