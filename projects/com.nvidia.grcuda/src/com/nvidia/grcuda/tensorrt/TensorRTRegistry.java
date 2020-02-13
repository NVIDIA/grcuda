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
package com.nvidia.grcuda.tensorrt;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.functions.FunctionTable;
import com.nvidia.grcuda.gpu.CUDAException;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedMessageException;

public class TensorRTRegistry {

    private static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    private static final String DEFAULT_LIBRARY = "libtrt.so";
    private static final String NAMESPACE = "TRT";
    private static final String TRT_ENABLED_PROPERTY_KEY = "tensorrt.enabled";
    private static final String TRT_LIBRARY_PROPERTY_KEY = "tensorrt.libpath";

    private final GrCUDAContext context;
    private final String libraryPath;

    public TensorRTRegistry(GrCUDAContext context) {
        this.context = context;
        libraryPath = System.getProperty(TRT_LIBRARY_PROPERTY_KEY, DEFAULT_LIBRARY);
        context.addDisposable(this::shutdown);
    }

    public void registerTensorRTFunctions(FunctionTable functionTable) {
        List<String> hiddenFunctions = Arrays.asList("enqueue");
        EnumSet.allOf(TensorRTFunctionNFI.class).stream().filter(func -> !hiddenFunctions.contains(func.getFunctionFactory().getName())).forEach(func -> {
            final ExternalFunctionFactory factory = func.getFunctionFactory();
            final Function nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath);
            Function function = nfiFunction;
            if (func.checkError) {
                // instantiate error checker decorator
                function = new Function(factory.getName(), NAMESPACE) {

                    @Override
                    @TruffleBoundary
                    public Object call(Object[] arguments) {
                        try {
                            Object result = INTEROP.execute(nfiFunction, arguments);
                            checkTRTReturnCode(result, nfiFunction.getName());
                            return result;
                        } catch (InteropException e) {
                            CompilerDirectives.transferToInterpreter();
                            throw new RuntimeException(e);
                        }
                    }
                };
            }
            functionTable.registerFunction(function);
        });
        functionTable.registerFunction(
                        new EnqueueFunction(TensorRTFunctionNFI.TRT_ENQUEUE.factory.makeFunction(context.getCUDARuntime(), libraryPath)));
    }

    private void shutdown() {

    }

    private static void checkTRTReturnCode(Object result, String function) {
        int returnCode;
        try {
            returnCode = INTEROP.asInt(result);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw new RuntimeException("expected return code as Integer object in " + function + ", got " + result.getClass().getName());
        }
        if (returnCode < 0) {
            CompilerDirectives.transferToInterpreter();
            throw new CUDAException(returnCode, trtReturnCodeToString(returnCode), function);
        }
    }

    private static String trtReturnCodeToString(int returnCode) {
        switch (returnCode) {
            case 0:
                return "TRT_OK";
            case -1:
                return "TRT_INVALID_HANDLE";
            case -2:
                return "TRT_UNABLE_TO_CREATE_RUNTIME";
            case -3:
                return "TRT_ENGINE_DESERIALIZATION_ERROR";
            case -4:
                return "TRT_ENGINE_FILE_NOT_FOUND";
            default:
                return "unknown error code: " + returnCode;
        }
    }

    public static boolean isTensorRTEnabled() {
        return System.getProperty(TRT_ENABLED_PROPERTY_KEY, "0").equals("1");
    }

    public enum TensorRTFunctionNFI {
        TRT_CREATE_INFER_RUNTIME(
                        new ExternalFunctionFactory("createInferRuntime",
                                        NAMESPACE, "createInferRuntime", "(): sint32"),
                        true),
        TRT_DESERIALIZE_CUDA_ENGINE(
                        new ExternalFunctionFactory("deserializeCudaEngine",
                                        NAMESPACE, "deserializeCudaEngine", "(sint32, string): sint32"),
                        true),
        TRT_DESTROY_INFER_RUNTIME(
                        new ExternalFunctionFactory("destroyInferRuntime",
                                        NAMESPACE, "destroyInferRuntime", "(sint32): sint32"),
                        true),
        TRT_CREATE_EXECUTION_CONTEXT(
                        new ExternalFunctionFactory("createExecutionContext",
                                        NAMESPACE, "createExecutionContext", "(sint32): sint32"),
                        true),
        TRT_GET_BINDING_INDEX(
                        new ExternalFunctionFactory("getBindingIndex",
                                        NAMESPACE, "getBindingIndex", "(sint32, string): sint32"),
                        false),
        TRT_GET_MAX_BATCH_SIZE(
                        new ExternalFunctionFactory("getMaxBatchSize",
                                        NAMESPACE, "getMaxBatchSize", "(sint32): sint32"),
                        false),
        TRT_ENQUEUE(
                        new ExternalFunctionFactory("enqueue",
                                        NAMESPACE, "enqueue", "(sint32, sint32, pointer, pointer, pointer): sint32"),
                        false),
        TRT_DESTROY_ENGINE(
                        new ExternalFunctionFactory("destroyEngine",
                                        NAMESPACE, "destroyEngine", "(sint32): sint32"),
                        true),
        TRT_DESTROY_EXECUTION_CONTEXT(
                        new ExternalFunctionFactory("destroyExecutionContext",
                                        NAMESPACE, "destroyExecutionContext", "(sint32): sint32"),
                        true);

        private final ExternalFunctionFactory factory;
        private final boolean checkError;

        public ExternalFunctionFactory getFunctionFactory() {
            return factory;
        }

        TensorRTFunctionNFI(ExternalFunctionFactory functionFactory, boolean checkError) {
            this.factory = functionFactory;
            this.checkError = checkError;
        }
    }
}
