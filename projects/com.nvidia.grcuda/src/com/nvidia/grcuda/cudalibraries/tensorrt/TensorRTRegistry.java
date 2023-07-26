/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.
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
package com.nvidia.grcuda.cudalibraries.tensorrt;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.GPUPointer;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.Namespace;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.GrCUDAException;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class TensorRTRegistry {

    private static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    public static final String DEFAULT_LIBRARY = "libtrt.so";
    public static final String NAMESPACE = "TRT";
    public static final String DEFAULT_LIBRARY_HINT = " (TensorRT library location can be set via the --grcuda.TensorRTLibrary= option. " +
                    "TensorRT support can be disabled via --grcuda.TensorRTEnabled=false.";

    private final GrCUDAContext context;
    private final String libraryPath;

    public TensorRTRegistry(GrCUDAContext context) {
        this.context = context;
        libraryPath = context.getOptions().getTensorRTLibrary();
        context.addDisposable(this::shutdown);
    }

    public void registerTensorRTFunctions(Namespace namespace) {
        List<String> hiddenFunctions = Arrays.asList("enqueue");
        EnumSet.allOf(TensorRTFunctionNFI.class).stream().filter(func -> !hiddenFunctions.contains(func.getFunctionFactory().getName())).forEach(func -> {
            final ExternalFunctionFactory factory = func.getFunctionFactory();
            Function function = (func.checkError) ? new ErrorCheckedTRTFunction(factory) : new TRTFunction(factory);
            namespace.addFunction(function);
        });
        namespace.addFunction(new EnqueueFunction(TensorRTFunctionNFI.TRT_ENQUEUE.factory));
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
            throw new GrCUDAException(returnCode, trtReturnCodeToString(returnCode), new String[]{function});
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

    public enum TensorRTFunctionNFI {
        TRT_CREATE_INFER_RUNTIME(
                        new ExternalFunctionFactory("createInferRuntime", "createInferRuntime", "(): sint32"),
                        true),
        TRT_DESERIALIZE_CUDA_ENGINE(
                        new ExternalFunctionFactory("deserializeCudaEngine", "deserializeCudaEngine", "(sint32, string): sint32"),
                        true),
        TRT_DESTROY_INFER_RUNTIME(
                        new ExternalFunctionFactory("destroyInferRuntime", "destroyInferRuntime", "(sint32): sint32"),
                        true),
        TRT_CREATE_EXECUTION_CONTEXT(
                        new ExternalFunctionFactory("createExecutionContext", "createExecutionContext", "(sint32): sint32"),
                        true),
        TRT_GET_BINDING_INDEX(
                        new ExternalFunctionFactory("getBindingIndex", "getBindingIndex", "(sint32, string): sint32"),
                        false),
        TRT_GET_MAX_BATCH_SIZE(
                        new ExternalFunctionFactory("getMaxBatchSize", "getMaxBatchSize", "(sint32): sint32"),
                        false),
        TRT_ENQUEUE(
                        new ExternalFunctionFactory("enqueue", "enqueue", "(sint32, sint32, pointer, pointer, pointer): sint32"),
                        false),
        TRT_DESTROY_ENGINE(
                        new ExternalFunctionFactory("destroyEngine", "destroyEngine", "(sint32): sint32"),
                        true),
        TRT_DESTROY_EXECUTION_CONTEXT(
                        new ExternalFunctionFactory("destroyExecutionContext", "destroyExecutionContext", "(sint32): sint32"),
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

    class TRTFunction extends Function {

        private final ExternalFunctionFactory factory;
        private Function nfiFunction;

        TRTFunction(ExternalFunctionFactory factory) {
            super(factory.getName());
            this.factory = factory;
        }

        @Override
        @TruffleBoundary
        public Object call(Object[] arguments) {
            try {
                if (nfiFunction == null) {
                    // load function symbol lazily
                    CompilerDirectives.transferToInterpreterAndInvalidate();
                    nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                }
                return INTEROP.execute(nfiFunction, arguments);
            } catch (InteropException e) {
                CompilerDirectives.transferToInterpreter();
                throw new RuntimeException(e);
            }
        }
    }

    class ErrorCheckedTRTFunction extends Function {
        private final ExternalFunctionFactory factory;
        private Function nfiFunction;

        ErrorCheckedTRTFunction(ExternalFunctionFactory factory) {
            super(factory.getName());
            this.factory = factory;
        }

        @Override
        @TruffleBoundary
        public Object call(Object[] arguments) {
            try {
                if (nfiFunction == null) {
                    // load function symbol lazily
                    CompilerDirectives.transferToInterpreterAndInvalidate();
                    nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                }
                Object result = INTEROP.execute(nfiFunction, arguments);
                checkTRTReturnCode(result, nfiFunction.getName());
                return result;
            } catch (InteropException e) {
                CompilerDirectives.transferToInterpreter();
                throw new RuntimeException(e);
            }
        }
    }

    class EnqueueFunction extends Function {
        private final ExternalFunctionFactory factory;
        private Function nfiFunction;

        protected EnqueueFunction(ExternalFunctionFactory factory) {
            super(factory.getName());
            this.factory = factory;
        }

        @Override
        protected Object call(Object[] arguments) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
            checkArgumentLength(arguments, 3);
            int engineHandle = expectInt(arguments[0]);
            int batchSize = expectInt(arguments[1]);

            // extract pointers from buffers array argument
            Object bufferArg = arguments[2];
            if (!INTEROP.hasArrayElements(bufferArg)) {
                throw UnsupportedMessageException.create();
            }
            int numBuffers = (int) INTEROP.getArraySize(bufferArg);
            try (UnsafeHelper.PointerArray pointerArray = UnsafeHelper.createPointerArray(numBuffers)) {
                if (nfiFunction == null) {
                    // load function symbol lazily
                    CompilerDirectives.transferToInterpreterAndInvalidate();
                    nfiFunction = factory.makeFunction(context.getCUDARuntime(), libraryPath, DEFAULT_LIBRARY_HINT);
                }
                for (int i = 0; i < numBuffers; ++i) {
                    try {
                        Object buffer = INTEROP.readArrayElement(bufferArg, i);
                        if (!(buffer instanceof DeviceArray) && !(buffer instanceof GPUPointer)) {
                            UnsupportedTypeException.create(new Object[]{buffer});
                        }
                        pointerArray.setValueAt(i, INTEROP.asPointer(buffer));
                    } catch (InvalidArrayIndexException e) {
                        InvalidArrayIndexException.create(i);
                    }
                }
                long stream = 0;
                long eventConsumed = 0;
                Object result = INTEROP.execute(nfiFunction, engineHandle, batchSize, pointerArray.getAddress(), stream, eventConsumed);
                if (!INTEROP.fitsInInt(result)) {
                    CompilerDirectives.transferToInterpreter();
                    throw new RuntimeException("result of 'enqueue' is not an int");
                }
                return INTEROP.asInt(result) == 1;
            }
        }
    }
}
