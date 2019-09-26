/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.nvidia.grcuda.cuml;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.functions.FunctionTable;
import com.nvidia.grcuda.gpu.CUDAException;
import com.nvidia.grcuda.gpu.UnsafeHelper;
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

    private static final String DEFAULT_LIBRARY = "libcuml.so";
    private static final String NAMESPACE = "ML";
    private static final String CUML_ENABLED_PROPERTY_KEY = "rapidsai.cuml.enabled";
    private static final String CUML_LIBRARY_PROPERTY_KEY = "rapidsai.cuml.libpath";

    private final GrCUDAContext context;
    private final String libraryPath;

    @CompilationFinal private TruffleObject cumlCreateFunction;

    @CompilationFinal private TruffleObject cumlDestroyFunction;

    @CompilationFinal private TruffleObject cumlCreateFunctionNFI;

    @CompilationFinal private TruffleObject cumlDestroyFunctionNFI;

    private Integer cumlHandle = null;

    public CUMLRegistry(GrCUDAContext context) {
        this.context = context;
        libraryPath = System.getProperty(CUML_LIBRARY_PROPERTY_KEY, DEFAULT_LIBRARY);
        context.addDisposable(this::cuMLShutdown);
    }

    public void registerCUMLFunctions(FunctionTable functionTable) {
        // create NFI function objects for handle creation and destruction
        cumlCreateFunctionNFI = CUMLFunctionNFI.CUML_CUMLCREATE.getFunctionFactory().makeFunction(
                        context.getCUDARuntime(), libraryPath);
        cumlDestroyFunctionNFI = CUMLFunctionNFI.CUML_CUMLDESTROY.getFunctionFactory().makeFunction(
                        context.getCUDARuntime(), libraryPath);

        // create wrapper for cumlCreate: cumlError_t cumlCreate(int* handle) -> int cumlCreate()
        cumlCreateFunction = new Function(
                        CUMLFunctionNFI.CUML_CUMLCREATE.getFunctionFactory().getName(), NAMESPACE) {
            @Override
            @TruffleBoundary
            public Object call(Object[] arguments) throws ArityException {
                checkArgumentLength(arguments, 0);
                try {
                    try (UnsafeHelper.Integer32Object handle = UnsafeHelper.createInteger32Object()) {
                        Object result = INTEROP.execute(cumlCreateFunctionNFI, handle.getAddress());
                        checkCUMLReturnCode(result, "cumlCreate");
                        return handle.getValue();
                    }
                } catch (InteropException e) {
                    CompilerDirectives.transferToInterpreter();
                    throw new RuntimeException(e);
                }
            }
        };

        // create wrapper for cumlDestroy: cumlError_t cumlDestroy(int handle) -> void
        // cumlDestroy(int handle)
        cumlDestroyFunction = new Function(
                        CUMLFunctionNFI.CUML_CUMLDESTROY.getFunctionFactory().getName(), NAMESPACE) {
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
                    throw new RuntimeException(e);
                }
            }
        };

        // Create function wrappers (decorators for all functions except handle con- and
        // destruction)
        List<String> hiddenFunctions = Arrays.asList("cumlCreate", "cumlDestroy");
        EnumSet.allOf(CUMLFunctionNFI.class).stream().filter(func -> !hiddenFunctions.contains(func.getFunctionFactory().getName())).forEach(func -> {
            final ExternalFunctionFactory factory = func.getFunctionFactory();
            final Function nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath);
            final Function wrapperFunction = new Function(factory.getName(), NAMESPACE) {

                @Override
                @TruffleBoundary
                public Object call(Object[] arguments) {
                    try {
                        if (cumlHandle == null) {
                            Object result = INTEROP.execute(cumlCreateFunction);
                            cumlHandle = expectInt(result);
                        }
                    } catch (InteropException e) {
                        CompilerDirectives.transferToInterpreter();
                        throw new RuntimeException(e);
                    }

                    // Argument 0 is the function name in the frame, removing argument 0 and
                    // replacing
                    // it with the handle argument does not change the size of the argument array.
                    Object[] argsWithHandle = new Object[arguments.length + 1];
                    System.arraycopy(arguments, 0, argsWithHandle, 1, arguments.length);
                    argsWithHandle[0] = cumlHandle;
                    try {
                        Object result = INTEROP.execute(nfiFunction, argsWithHandle);
                        checkCUMLReturnCode(result, nfiFunction.getName());
                        return result;
                    } catch (InteropException e) {
                        CompilerDirectives.transferToInterpreter();
                        throw new RuntimeException(e);
                    }
                }
            };
            functionTable.registerFunction(wrapperFunction);
        });
    }

    private void cuMLShutdown() {
        if (cumlHandle != null) {
            try {
                Object result = INTEROP.execute(cumlDestroyFunction, cumlHandle);
                checkCUMLReturnCode(result, CUMLFunctionNFI.CUML_CUMLDESTROY.getFunctionFactory().getName());
            } catch (InteropException e) {
                CompilerDirectives.transferToInterpreter();
                throw new RuntimeException(e);
            }
        }
    }

    private static void checkCUMLReturnCode(Object result, String function) {
        int returnCode;
        try {
            returnCode = INTEROP.asInt(result);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw new RuntimeException("expected return code as Integer object in " + function + ", got " + result.getClass().getName());
        }
        if (returnCode != 0) {
            CompilerDirectives.transferToInterpreter();
            throw new CUDAException(returnCode, cumlReturnCodeToString(returnCode), function);
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

    public static boolean isCUMLEnabled() {
        return System.getProperty(CUML_ENABLED_PROPERTY_KEY) != null;
    }

    public enum CUMLFunctionNFI {
        CUML_CUMLCREATE(
                        new ExternalFunctionFactory("cumlCreate",
                                        NAMESPACE, "cumlCreate", "(pointer): sint32")),
        CUML_CUMLDESTROY(
                        new ExternalFunctionFactory("cumlDestroy",
                                        NAMESPACE, "cumlDestroy", "(sint32): sint32")),
        CUML_DBSCANFITDOUBLE(
                        new ExternalFunctionFactory("cumlDpDbscanFit",
                                        NAMESPACE, "cumlDpDbscanFit",
                                        "(sint32, pointer, sint32, sint32, double, sint32, pointer, uint64, sint32): sint32")),
        CUML_DBSCANFITFLOAT(
                        new ExternalFunctionFactory("cumlSpDbscanFit",
                                        NAMESPACE, "cumlSpDbscanFit",
                                        "(sint32, pointer, sint32, sint32, float, sint32, pointer, uint64, sint32): sint32"));

        private final ExternalFunctionFactory factory;

        public ExternalFunctionFactory getFunctionFactory() {
            return factory;
        }

        CUMLFunctionNFI(ExternalFunctionFactory functionFactory) {
            this.factory = functionFactory;
        }
    }
}
