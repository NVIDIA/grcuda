/*
 * Copyright (c) 2021, NECSTLab, Politecnico di Milano. All rights reserved.
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
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
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
package com.nvidia.grcuda.cudalibraries.cusparse;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectLong;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.Namespace;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.cudalibraries.CUDALibraryFunction;
import com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy.CUSPARSEProxy;
import com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy.CUSPARSEProxyGemvi;
import com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy.CUSPARSEProxySpMV;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.computation.CUDALibraryExecution;
import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.stream.CUSPARSESetStreamFunction;
import com.nvidia.grcuda.runtime.stream.LibrarySetStreamFunction;
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

public class CUSPARSERegistry {
    public static final String DEFAULT_LIBRARY = (System.getenv("LIBCUSPARSE_DIR") != null ? System.getenv("LIBCUSPARSE_DIR") : "") + "libcusparse.so";
    public static final String DEFAULT_LIBRARY_HINT = " (CuSPARSE library location can be set via the --grcuda.CuSPARSELibrary= option. " +
                    "CuSPARSE support can be disabled via --grcuda.CuSPARSEEnabled=false.";
    public static final String NAMESPACE = "SPARSE";

    private final GrCUDAContext context;
    private final String libraryPath;

    private LibrarySetStreamFunction cusparseLibrarySetStreamFunction;

    @CompilationFinal private TruffleObject cusparseCreateFunction;
    @CompilationFinal private TruffleObject cusparseDestroyFunction;
    @CompilationFinal private TruffleObject cusparseSetStreamFunction;

    @CompilationFinal private TruffleObject cusparseCreateFunctionNFI;
    @CompilationFinal private TruffleObject cusparseDestroyFunctionNFI;
    @CompilationFinal private TruffleObject cusparseSetStreamFunctionNFI;


    private Long cusparseHandle = null;

    public enum CUSPARSEIndexType {
        CUSPARSE_INDEX_UNUSED,
        CUSPARSE_INDEX_16U,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_64I;
    }

    public enum CUSPARSEIndexBase {
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_INDEX_BASE_ONE;
    }

    public enum CUDADataType {
        CUDA_R_32F, // 32 bit real
        CUDA_R_64F, // 64 bit real
        CUDA_R_16F, // 16 bit real
        CUDA_R_8I, // 8 bit real as a signed integer
        CUDA_C_32F, // 32 bit complex
        CUDA_C_64F, // 64 bit complex
        CUDA_C_16F, // 16 bit complex
        CUDA_C_8I,  // 8 bit complex as a pair of signed integers
        CUDA_R_8U,   // 8 bit real as a signed integer
        CUDA_C_8U;  // 8 bit complex as a pair of signed integers
    }

    public enum CUSPARSEOperation {
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    }

    public enum CUSPARSESpMVAlg {
        CUSPARSE_SPMV_ALG_DEFAULT,
        CUSPARSE_SPMV_COO_ALG1,
        CUSPARSE_SPMV_COO_ALG2,
        CUSPARSE_SPMV_CSR_ALG1,
        CUSPARSE_SPMV_CSR_ALG2;
    }

    public CUSPARSERegistry(GrCUDAContext context) {
        this.context = context;
        // created field in GrCUDAOptions
        libraryPath = context.getOptions().getCuSPARSELibrary();
    }

    public void ensureInitialized() {
        if (cusparseHandle == null) {

            CUSPARSEProxy.setContext(context);

            CompilerDirectives.transferToInterpreterAndInvalidate();

            // create NFI function objects for functions' management

            cusparseCreateFunctionNFI = CUSPARSE_CUSPARSECREATE.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseDestroyFunctionNFI = CUSPARSE_CUSPARSEDESTROY.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cusparseSetStreamFunctionNFI = CUSPARSE_CUSPARSESETSTREAM.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);

            // cusparseStatus_t cusparseCreate(cusparseHandle_t handle)

            cusparseCreateFunction = new Function(CUSPARSE_CUSPARSECREATE.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException {
                    checkArgumentLength(arguments, 0);
                    try (UnsafeHelper.Integer64Object handle = UnsafeHelper.createInteger64Object()) {
                        Object result = INTEROP.execute(cusparseCreateFunctionNFI, handle.getAddress());
                        checkCUSPARSEReturnCode(result, "cusparseCreate");
                        return handle.getValue();
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // cusparseStatus_t cusparseDestroy(cusparseHandle_t* handle)

            cusparseDestroyFunction = new Function(CUSPARSE_CUSPARSEDESTROY.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 1);
                    long handle = expectLong(arguments[0]);
                    try {
                        Object result = INTEROP.execute(cusparseDestroyFunctionNFI, handle);
                        checkCUSPARSEReturnCode(result, "cusparseDestroy");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId)

            cusparseSetStreamFunction = new Function(CUSPARSE_CUSPARSESETSTREAM.getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 2);
                    long handle = expectLong(arguments[0]);
                    long streamId = expectLong(arguments[1]);
                    try {
                        Object result = INTEROP.execute(cusparseSetStreamFunctionNFI, handle, streamId);
                        checkCUSPARSEReturnCode(result, "cusparseSetStream");
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            try {
                Object result = INTEROP.execute(cusparseCreateFunction);
                cusparseHandle = expectLong(result);
                context.addDisposable(this::cuSPARSEShutdown);
            } catch (InteropException e) {
                throw new GrCUDAInternalException(e);
            }
        }

        cusparseLibrarySetStreamFunction = new CUSPARSESetStreamFunction((Function) cusparseSetStreamFunctionNFI, cusparseHandle);

    }

    private void cuSPARSEShutdown() {
        CompilerAsserts.neverPartOfCompilation();
        if (cusparseHandle != null) {
            try {
                Object result = InteropLibrary.getFactory().getUncached().execute(cusparseDestroyFunction, cusparseHandle);
                checkCUSPARSEReturnCode(result, CUSPARSE_CUSPARSEDESTROY.getName());
                cusparseHandle = null;
            } catch (InteropException e) {
                throw new GrCUDAInternalException(e);
            }
        }
    }

    public void registerCUSPARSEFunctions(Namespace namespace) {
        // Create function wrappers
        for (CUSPARSEProxy proxy : functions) {
            final Function wrapperFunction = new CUDALibraryFunction(proxy.getExternalFunctionFactory().getName(), proxy.getExternalFunctionFactory().getNFISignature()) {

                private Function nfiFunction;

                @Override
                public List<ComputationArgumentWithValue> createComputationArgumentWithValueList(Object[] args, Long libraryHandle) {
                    ArrayList<ComputationArgumentWithValue> argumentsWithValue = new ArrayList<>();
                    // Set the library handle;
                    argumentsWithValue.add(new ComputationArgumentWithValue(this.computationArguments.get(0), libraryHandle));
                    // Set the other arguments (size - 1 as we skip the handle, i.e. the argument 0);
                    for (int i = 0; i < this.computationArguments.size() - 1; i++) {
                        argumentsWithValue.add(new ComputationArgumentWithValue(this.computationArguments.get(i + 1), args[i]));
                    }
                    // Add extra arguments at the end: they are used to track input DeviceArrays;
                    int numExtraArrays = args.length - (this.computationArguments.size() - 1);
                    for (int i = 0; i < numExtraArrays; i++) {
                        argumentsWithValue.add(new ComputationArgumentWithValue(
                                "cusparse_extra_array_" + i, Type.NFI_POINTER, ComputationArgument.Kind.POINTER_INOUT,
                                args[this.computationArguments.size() - 1 + i]));
                    }
                    return argumentsWithValue;
                }

                @Override
                @TruffleBoundary
                protected Object call(Object[] arguments) {
                    ensureInitialized();

                    try {
                        if (nfiFunction == null) {
                            CompilerDirectives.transferToInterpreterAndInvalidate();
                            nfiFunction = proxy.getExternalFunctionFactory().makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                        }
                        // This list of arguments might have extra arguments: the DeviceArrays that can cause dependencies but are not directly used by the cuSPARSE function,
                        //   as these DeviceArrays might be wrapped using cuSparseMatrices/Vectors/Buffers.
                        // We still need to pass these DeviceArrays to the GrCUDAComputationalElement so we track dependencies correctly,
                        // but they are removed from the final list of arguments passed to the cuSPARSE library;
                        Object[] formattedArguments = proxy.formatArguments(arguments, cusparseHandle);
                        List<ComputationArgumentWithValue> computationArgumentsWithValue = this.createComputationArgumentWithValueList(formattedArguments, cusparseHandle);
                        int extraArraysToTrack = computationArgumentsWithValue.size() - this.computationArguments.size();  // Both lists also contain the handle;
                        Object result = new CUDALibraryExecution(context.getGrCUDAExecutionContext(), nfiFunction, cusparseLibrarySetStreamFunction,
                                computationArgumentsWithValue, extraArraysToTrack).schedule();

                        checkCUSPARSEReturnCode(result, nfiFunction.getName());
                        return result;
                    } catch (InteropException e) {
                        throw new GrCUDAInternalException(e);
                    }
                }
            };
            namespace.addFunction(wrapperFunction);
        }
    }

    public static void checkCUSPARSEReturnCode(Object result, String... function) {
        CompilerAsserts.neverPartOfCompilation();
        int returnCode;
        try {
            returnCode = InteropLibrary.getFactory().getUncached().asInt(result);
        } catch (UnsupportedMessageException e) {
            throw new GrCUDAInternalException("expected return code as Integer object in " + Arrays.toString(function) + ", got " + result.getClass().getName());
        }
        if (returnCode != 0) {
            throw new GrCUDAException(returnCode, cusparseReturnCodeToString(returnCode), function);
        }
    }

    public static String cusparseReturnCodeToString(int returnCode) {
        switch (returnCode) {
            case 0:
                return "CUSPARSE_STATUS_SUCCESS";
            case 1:
                return "CUSPARSE_STATUS_NOT_INITIALIZED";
            case 2:
                return "CUSPARSE_STATUS_ALLOC_FAILED";
            case 3:
                return "CUSPARSE_STATUS_INVALID_VALUE";
            case 4:
                return "CUSPARSE_STATUS_ARCH_MISMATCH";
            case 5:
                return "CUSPARSE_STATUS_EXECUTION_FAILED";
            case 6:
                return "CUSPARSE_STATUS_INTERNAL_ERROR";
            case 7:
                return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            case 8:
                return "CUSPARSE_STATUS_NOT_SUPPORTED";
            case 9:
                return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
            default:
                return "unknown error code: " + returnCode;
        }
    }

    // functions exposed to the user

    private static final ExternalFunctionFactory CUSPARSE_CUSPARSECREATE = new ExternalFunctionFactory("cusparseCreate", "cusparseCreate", "(pointer): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSEDESTROY = new ExternalFunctionFactory("cusparseDestroy", "cusparseDestroy", "(sint64): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESETSTREAM = new ExternalFunctionFactory("cusparseSetStream", "cusparseSetStream", "(sint64, sint64): sint32");
    private static final ExternalFunctionFactory CUSPARSE_CUSPARSESPMV = new ExternalFunctionFactory("cusparseSpMV", "cusparseSpMV", "(sint64, sint32, pointer, sint64, " +
                    "sint64, pointer, sint64, sint32, sint32, pointer): sint32");
    private static final ArrayList<CUSPARSEProxy> functions = new ArrayList<>();

    static {

        for (char type : new char[]{'S', 'D', 'C', 'Z'}) {
            final ExternalFunctionFactory CUSPARSE_CUSPARSEGEMVI = new ExternalFunctionFactory("cusparse" + type + "gemvi", "cusparseSgemvi", "(sint64, sint32, sint32, sint32," +
                    "pointer, pointer, sint32, sint32, pointer, pointer, pointer, pointer, sint32, pointer): sint32");
            functions.add(new CUSPARSEProxyGemvi(CUSPARSE_CUSPARSEGEMVI));
        }

        functions.add(new CUSPARSEProxySpMV(CUSPARSE_CUSPARSESPMV));
    }

}
