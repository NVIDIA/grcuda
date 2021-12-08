package com.nvidia.grcuda.runtime.stream;

import static com.nvidia.grcuda.functions.Function.INTEROP;

import com.nvidia.grcuda.functions.Function;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

/**
 * Class of functions to avoid managing streams in the CUSPARSE library
 */

public class CUSPARSESetStreamFunction extends LibrarySetStreamFunction {

    private final long handle;

    public CUSPARSESetStreamFunction(Function setStreamFunctionNFI, long handle) {
        super(setStreamFunctionNFI);
        this.handle = handle;
    }

    @Override
    public void setStream(CUDAStream stream) {
        Object[] cusparseSetStreamArgs = {this.handle, stream.getRawPointer()};
        try {
            INTEROP.execute(this.setStreamFunctionNFI, cusparseSetStreamArgs);
        } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
            System.out.println("failed to set CUSPARSE stream");
            e.printStackTrace();
        }
    }
}