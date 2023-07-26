package com.nvidia.grcuda.runtime.stream;

import com.nvidia.grcuda.functions.Function;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import static com.nvidia.grcuda.functions.Function.INTEROP;

/**
 * Class of functions to manage streams in the CUML library
 */

public class LibrarySetStreamCUML extends LibrarySetStream {

    private final long handle;

    public LibrarySetStreamCUML(Function setStreamFunctionNFI, long handle) {
        super(setStreamFunctionNFI);
        this.handle = handle;
    }

    @Override
    public void setStream(CUDAStream stream) {
        Object[] cumlSetStreamArgs = {this.handle, stream.getRawPointer()};
        try {
            INTEROP.execute(this.setStreamFunctionNFI, cumlSetStreamArgs);
        } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("failed to set CUML stream");
            e.printStackTrace();
        }
    }
}
