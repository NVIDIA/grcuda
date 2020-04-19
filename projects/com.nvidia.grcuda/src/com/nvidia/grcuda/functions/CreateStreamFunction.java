package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CreateStreamFunction extends Function {
    private final CUDARuntime cudaRuntime;

    public CreateStreamFunction(CUDARuntime cudaRuntime) {
        super("createstream");
        this.cudaRuntime = cudaRuntime;
    }

    @Override
    @CompilerDirectives.TruffleBoundary
    public Object call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        checkArgumentLength(arguments, 0);
        return cudaRuntime.cudaStreamCreate();
    }
}
