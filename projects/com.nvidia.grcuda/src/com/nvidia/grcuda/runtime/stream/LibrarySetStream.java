package com.nvidia.grcuda.runtime.stream;

import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

/**
 * Abstract class to manage async streams for supported libraries
 */

abstract public class LibrarySetStream {

    protected final Function setStreamFunctionNFI;

    protected LibrarySetStream(Function setStreamFunctionNFI) {
        this.setStreamFunctionNFI = setStreamFunctionNFI;
    }

    /**
     * Set stream for the execution of supported libraries' functions
     * 
     * @param stream a CUDAstream
     */
    public abstract void setStream(CUDAStream stream);
}
