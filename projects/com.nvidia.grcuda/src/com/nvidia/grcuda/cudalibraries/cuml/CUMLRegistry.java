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
package com.nvidia.grcuda.cudalibraries.cuml;

import static com.nvidia.grcuda.functions.Function.expectInt;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.Namespace;
import com.nvidia.grcuda.cudalibraries.CUDALibraryFunction;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.computation.CUDALibraryExecution;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.CompilationFinal;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUMLRegistry {

    private static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    public static final String DEFAULT_LIBRARY = (System.getenv("LIBCUML_DIR") != null ? System.getenv("LIBCUML_DIR") : "") + "libcuml.so";

    public static final String DEFAULT_LIBRARY_HINT = " (CuML library location can be set via the --grcuda.CuMLLibrary= option. " +
            "CuML support can be disabled via --grcuda.CuMLEnabled=false.";
    public static final String NAMESPACE = "ML";

    private final GrCUDAContext context;
    private final String libraryPath;

    @CompilationFinal private TruffleObject cumlCreateFunction;

    @CompilationFinal private TruffleObject cumlDestroyFunction;

    @CompilationFinal private TruffleObject cumlCreateFunctionNFI;

    @CompilationFinal private TruffleObject cumlDestroyFunctionNFI;

    private Integer cumlHandle = null;

    public CUMLRegistry(GrCUDAContext context) {
        this.context = context;
        libraryPath = context.getOption(GrCUDAOptions.CuMLLibrary);
    }

    private void ensureInitialized() {
        if (cumlHandle == null) {
            CompilerDirectives.transferToInterpreterAndInvalidate();

            // create NFI function objects for handle creation and destruction
            cumlCreateFunctionNFI = CUMLFunctionNFI.CUML_CUMLCREATE.getFunctionFactory().makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
            cumlDestroyFunctionNFI = CUMLFunctionNFI.CUML_CUMLDESTROY.getFunctionFactory().makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);

            // create wrapper for cumlCreate: cumlError_t cumlCreate(int* handle) -> int
            // cumlCreate()
            cumlCreateFunction = new Function(CUMLFunctionNFI.CUML_CUMLCREATE.getFunctionFactory().getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException {
                    checkArgumentLength(arguments, 0);
                    try (UnsafeHelper.Integer32Object handle = UnsafeHelper.createInteger32Object()) {
                        Object result = INTEROP.execute(cumlCreateFunctionNFI, handle.getAddress());
                        checkCUMLReturnCode(result, "cumlCreate");
                        return handle.getValue();
                    } catch (InteropException e) {
                        CompilerDirectives.transferToInterpreter();
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            // create wrapper for cumlDestroy: cumlError_t cumlDestroy(int handle) -> void
            // cumlDestroy(int handle)
            cumlDestroyFunction = new Function(CUMLFunctionNFI.CUML_CUMLDESTROY.getFunctionFactory().getName()) {
                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
                    checkArgumentLength(arguments, 1);
                    Object handle = expectInt(arguments[0]);
                    try {
                        Object result = INTEROP.execute(cumlDestroyFunctionNFI, handle);
                        checkCUMLReturnCode(result, "cumlDestroy");
                        return result;
                    } catch (InteropException e) {
                        CompilerDirectives.transferToInterpreter();
                        throw new GrCUDAInternalException(e);
                    }
                }
            };

            try {
                Object result = INTEROP.execute(cumlCreateFunction);
                cumlHandle = expectInt(result);
                context.addDisposable(this::cuMLShutdown);
            } catch (InteropException e) {
                CompilerDirectives.transferToInterpreter();
                throw new GrCUDAInternalException(e);
            }
        }
    }

    private void cuMLShutdown() {
        if (cumlHandle != null) {
            try {
                Object result = INTEROP.execute(cumlDestroyFunction, cumlHandle);
                checkCUMLReturnCode(result, CUMLFunctionNFI.CUML_CUMLDESTROY.getFunctionFactory().getName());
                cumlHandle = null;
            } catch (InteropException e) {
                CompilerDirectives.transferToInterpreter();
                throw new GrCUDAInternalException(e);
            }
        }
    }

    public void registerCUMLFunctions(Namespace namespace) {
        // Create function wrappers (decorators for all functions except handle con- and
        // destruction)
        List<CUMLFunctionNFI> hiddenFunctions = Arrays.asList(CUMLFunctionNFI.CUML_CUMLCREATE, CUMLFunctionNFI.CUML_CUMLDESTROY);
        EnumSet.allOf(CUMLFunctionNFI.class).stream().filter(func -> !hiddenFunctions.contains(func)).forEach(func -> {
            final ExternalFunctionFactory factory = func.getFunctionFactory();
            final Function wrapperFunction = new CUDALibraryFunction(factory.getName(), factory.getNFISignature()) {

                private Function nfiFunction;

                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) {
                    ensureInitialized();

                    try {
                        if (nfiFunction == null) {
                            CompilerDirectives.transferToInterpreterAndInvalidate();
                            nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                        }
                        Object result = new CUDALibraryExecution(context.getGrCUDAExecutionContext(), nfiFunction, this.createComputationArgumentWithValueList(arguments, (long) cumlHandle)).schedule();
                        checkCUMLReturnCode(result, nfiFunction.getName());
                        return result;
                    } catch (InteropException e) {
                        CompilerDirectives.transferToInterpreter();
                        throw new GrCUDAInternalException(e);
                    }
                }
            };
            namespace.addFunction(wrapperFunction);
        });
    }

    private static void checkCUMLReturnCode(Object result, String... function) {
        int returnCode;
        try {
            returnCode = INTEROP.asInt(result);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAInternalException("expected return code as Integer object in " + GrCUDAException.format(function) + ", got " + result.getClass().getName());
        }
        if (returnCode != 0) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(returnCode, cumlReturnCodeToString(returnCode), function);
        }
    }

    private static String cumlReturnCodeToString(int returnCode) {
        switch (returnCode) {
            case 0:
                return "CUML_SUCCESS";
            case 1:
                return "CUML_ERROR_UNKNOWN";
            case 2:
                return "CUML_INVALID_HANDLE";
            default:
                return "unknown error code: " + returnCode;
        }
    }

    public enum CUMLFunctionNFI {
        CUML_CUMLCREATE(new ExternalFunctionFactory("cumlCreate", "cumlCreate", "(pointer): sint32")),
        CUML_CUMLDESTROY(new ExternalFunctionFactory("cumlDestroy", "cumlDestroy", "(sint32): sint32")),
        CUML_DBSCANFITDOUBLE(new ExternalFunctionFactory("cumlDpDbscanFit", "cumlDpDbscanFit", "(sint32, pointer, sint32, sint32, double, sint32, pointer, uint64, sint32): sint32")),
        CUML_DBSCANFITFLOAT(new ExternalFunctionFactory("cumlSpDbscanFit", "cumlSpDbscanFit", "(sint32, pointer, sint32, sint32, float, sint32, pointer, uint64, sint32): sint32"));

        private final ExternalFunctionFactory factory;

        public ExternalFunctionFactory getFunctionFactory() {
            return factory;
        }

        CUMLFunctionNFI(ExternalFunctionFactory functionFactory) {
            this.factory = functionFactory;
        }
    }
}
