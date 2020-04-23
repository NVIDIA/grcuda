package com.nvidia.grcuda.gpu.stream;

import com.nvidia.grcuda.GPUPointer;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.Objects;

@ExportLibrary(InteropLibrary.class)
public class CUDAStream extends GPUPointer {

    private final int streamNumber;

    public CUDAStream(long rawPointer, int streamNumber) {
        super(rawPointer);
        this.streamNumber = streamNumber;
    }

    public int getStreamNumber() {
        return streamNumber;
    }

    public boolean isDefaultStream() { return false; }

    @Override
    public String toString() {
        return "CUDAStream(streamNumber=" + this.streamNumber + "; address=0x" + Long.toHexString(this.getRawPointer()) + ")";
    }

    @ExportMessage
    public Object toDisplayString(boolean allowSideEffect) {
        return this.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CUDAStream that = (CUDAStream) o;
        return (streamNumber == that.streamNumber && this.getRawPointer() == that.getRawPointer());
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), streamNumber);
    }
}
