package com.nvidia.grcuda.runtime.stream;

public class DefaultStream extends CUDAStream {
    
    static final int DEFAULT_STREAM_NUMBER = -1;

    private static final DefaultStream defaultStream = new DefaultStream();
    
    private DefaultStream() {
        super(0, DEFAULT_STREAM_NUMBER);
    }

    public static DefaultStream get() { return defaultStream; }

    @Override
    public boolean isDefaultStream() {
        return true; }

    @Override
    public String toString() {
        return "DefaultCUDAStream(streamNumber=" + DEFAULT_STREAM_NUMBER + "; address=0x" + Long.toHexString(this.getRawPointer()) + ")";
    }
}
