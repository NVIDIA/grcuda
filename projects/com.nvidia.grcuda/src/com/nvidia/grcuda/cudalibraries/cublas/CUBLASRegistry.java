/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.nvidia.grcuda.cudalibraries.cublas;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectLong;

import java.util.ArrayList;
import java.util.Arrays;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.Namespace;
import com.nvidia.grcuda.cudalibraries.CUDALibraryFunction;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.gpu.UnsafeHelper;
import com.nvidia.grcuda.gpu.computation.CUDALibraryExecution;
import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.CompilationFinal;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUBLASRegistry {
    public static final String DEFAULT_LIBRARY = "libcublas.so";
    public static final String DEFAULT_LIBRARY_HINT = " (CuBLAS library location can be set via the --grcuda.CuBLASLibrary= option. " +
                    "CuBLAS support can be disabled via --grcuda.CuBLASEnabled=false.";
    public static final String NAMESPACE = "BLAS";

    private final GrCUDAContext context;
    private final String libraryPath;

    @CompilationFinal private TruffleObject cublasCreateFunction;
    @CompilationFinal private TruffleObject cublasDestroyFunction;
    @CompilationFinal private TruffleObject cublasCreateFunctionNFI;
    @CompilationFinal private TruffleObject cublasDestroyFunctionNFI;

    private Long cublasHandle = null;

    public CUBLASRegistry(GrCUDAContext context) {
        this.context = context;
        libraryPath = context.getOption(GrCUDAOptions.CuBLASLibrary);
    }

    public void ensureInitialized() {
        if (cublasHandle == null) {
            CompilerDirectives.transferToInterpreterAndInvalidate();

            // create NFI function objects for handle creation and destruction

            cublasCreateFunctionNFI = CUBLAS_CUBLASCREATE.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cublasDestroyFunctionNFI = CUBLAS_CUBLASDESTROY.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);

            // create wrapper for cublasCreate: cublasError_t cublasCreate(long* handle) -> int
            // cublasCreate()
            cublasCreateFunction = new Function(CUBLAS_CUBLASCREATE.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException {
                    checkArgumentLength(arguments, 0);
                    try (UnsafeHelper.Integer64Object handle = UnsafeHelper.createInteger64Object()) {
                        Object result = INTEROP.execute(cublasCreateFunctionNFI, handle.getAddress());
                        checkCUBLASReturnCode(result, "cublasCreate");
                        return handle.getValue();
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // create wrapper for cublasDestroy: cublasError_t cublasDestroy(long handle) -> void
            // cublasDestroy(long handle)
            cublasDestroyFunction = new Function(CUBLAS_CUBLASDESTROY.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 1);
                    long handle = expectLong(arguments[0]);
                    try {
                        Object result = INTEROP.execute(cublasDestroyFunctionNFI, handle);
                        checkCUBLASReturnCode(result, "cublasDestroy");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            try {
                Object result = INTEROP.execute(cublasCreateFunction);
                cublasHandle = expectLong(result);

                context.addDisposable(this::cuBLASShutdown);
            } catch (InteropException e) {
                throw new GrCUDAInternalException(e);
            }
        }
    }

    private void cuBLASShutdown() {
        CompilerAsserts.neverPartOfCompilation();
        if (cublasHandle != null) {
            try {
                Object result = InteropLibrary.getFactory().getUncached().execute(cublasDestroyFunction, cublasHandle);
                checkCUBLASReturnCode(result, CUBLAS_CUBLASDESTROY.getName());
                cublasHandle = null;
            } catch (InteropException e) {
                throw new GrCUDAInternalException(e);
            }
        }
    }

    public void registerCUBLASFunctions(Namespace namespace) {
        // Create function wrappers (decorators for all functions except handle con- and destruction);
        for (ExternalFunctionFactory factory : functions) {
            final Function wrapperFunction = new CUDALibraryFunction(factory.getName(), factory.getNFISignature()) {

                private Function nfiFunction;

                @Override
                @TruffleBoundary
                protected Object call(Object[] arguments) {
                    ensureInitialized();

                    try {
                        if (nfiFunction == null) {
                            CompilerDirectives.transferToInterpreterAndInvalidate();
                            nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                        }
                        Object result = new CUDALibraryExecution(context.getGrCUDAExecutionContext(), nfiFunction, this.createComputationArgumentWithValueList(arguments, cublasHandle)).schedule();
                        checkCUBLASReturnCode(result, nfiFunction.getName());
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };
            namespace.addFunction(wrapperFunction);
        }
    }

    private static void checkCUBLASReturnCode(Object result, String... function) {
        CompilerAsserts.neverPartOfCompilation();
        int returnCode;
        try {
            returnCode = InteropLibrary.getFactory().getUncached().asInt(result);
        } catch (UnsupportedMessageException e) {
            throw new GrCUDAInternalException("expected return code as Integer object in " + Arrays.toString(function) + ", got " + result.getClass().getName());
        }
        if (returnCode != 0) {
            throw new GrCUDAException(returnCode, cublasReturnCodeToString(returnCode), function);
        }
    }

    private static String cublasReturnCodeToString(int returnCode) {
        switch (returnCode) {
            case 0:
                return "CUBLAS_STATUS_SUCCESS";
            case 1:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case 3:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case 7:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case 8:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case 11:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case 13:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case 14:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case 15:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
            case 16:
                return "CUBLAS_STATUS_LICENSE_ERROR";
            default:
                return "unknown error code: " + returnCode;
        }
    }

    private static final ExternalFunctionFactory CUBLAS_CUBLASCREATE = new ExternalFunctionFactory("cublasCreate", "cublasCreate_v2", "(pointer): sint32");
    private static final ExternalFunctionFactory CUBLAS_CUBLASDESTROY = new ExternalFunctionFactory("cublasDestroy", "cublasDestroy_v2", "(sint64): sint32");

    private static final ArrayList<ExternalFunctionFactory> functions = new ArrayList<>();

    static {
        for (char type : new char[]{'S', 'D', 'C', 'Z'}) {
            functions.add(new ExternalFunctionFactory("cublas" + type + "axpy", "cublas" + type + "axpy_v2",
                            "(sint64, sint32, pointer, pointer, sint32, pointer, sint32): sint32"));
            functions.add(new ExternalFunctionFactory("cublas" + type + "gemv", "cublas" + type + "gemv_v2",
                            "(sint64, sint32, sint32, sint32, pointer, pointer, sint32, pointer, sint32, pointer, pointer, sint32): sint32"));
            functions.add(new ExternalFunctionFactory("cublas" + type + "gemm", "cublas" + type + "gemm_v2",
                            "(sint64, sint32, sint32, sint32, sint32, sint32, pointer, pointer, sint32, pointer, sint32, pointer, pointer, sint32): sint32"));
        }
    }
}
