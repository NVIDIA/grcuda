package com.nvidia.grcuda.gpu.stream;

public class DefaultStream extends CUDAStream {
    
    static final int DEFAULT_STREAM_NUMBER = 0;
    
    public DefaultStream() {
        super(0, DEFAULT_STREAM_NUMBER);
    }

    @Override
    public boolean isDefaultStream() { return true; }

    @Override
    public String toString() {
        return "DefaultCUDAStream(streamNumber=" + DEFAULT_STREAM_NUMBER + "; address=0x" + Long.toHexString(this.getRawPointer()) + ")";
    }
}
