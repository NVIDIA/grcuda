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
package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectInt;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUSPARSEProxyGemvi extends CUSPARSEProxy {

    private final int N_ARGS_RAW = 13; // args for library function

    public CUSPARSEProxyGemvi(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs, long handle) throws UnsupportedTypeException, UnsupportedMessageException, ArityException {
        this.initializeNfi();
        if (rawArgs.length == 0) {
            return rawArgs;
        } else {

            args = new Object[N_ARGS_RAW];

            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();

            bufferSize.setValue(0);

            CUSPARSERegistry.CUSPARSEOperation transA = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(rawArgs[0])];
            int rows = expectInt(rawArgs[1]);
            int cols = expectInt(rawArgs[2]);
            DeviceArray alpha = (DeviceArray) rawArgs[3];
            DeviceArray matA = (DeviceArray) rawArgs[4];
            int lda = expectInt(rawArgs[5]);
            int nnz = expectInt(rawArgs[6]);
            DeviceArray x = (DeviceArray) rawArgs[7];
            DeviceArray xInd = (DeviceArray) rawArgs[8];
            DeviceArray beta = (DeviceArray) rawArgs[9];
            DeviceArray outVec = (DeviceArray) rawArgs[10];
            CUSPARSERegistry.CUSPARSEIndexBase idxBase = CUSPARSERegistry.CUSPARSEIndexBase.values()[expectInt(rawArgs[11])];
            char type = (char) rawArgs[12];

            // create buffer

            switch (type) {
                case 'S': {
                    Object resultBufferSize = INTEROP.execute(cusparseSgemvi_bufferSizeFunction, handle, transA.ordinal(), rows, cols, nnz, bufferSize.getAddress());
                    CUSPARSERegistry.checkCUSPARSEReturnCode(resultBufferSize, cusparseSgemvi_bufferSizeFunction.toString());
                    break;
                }
                case 'D': {
                    Object resultBufferSize = INTEROP.execute(cusparseDgemvi_bufferSizeFunction, handle, transA.ordinal(), rows, cols, nnz, bufferSize.getAddress());
                    CUSPARSERegistry.checkCUSPARSEReturnCode(resultBufferSize, cusparseDgemvi_bufferSizeFunction.toString());
                    break;
                }
                case 'C': {
                    Object resultBufferSize = INTEROP.execute(cusparseCgemvi_bufferSizeFunction, handle, transA.ordinal(), rows, cols, nnz, bufferSize.getAddress());
                    CUSPARSERegistry.checkCUSPARSEReturnCode(resultBufferSize, cusparseCgemvi_bufferSizeFunction.toString());
                    break;
                }
                case 'Z': {
                    Object resultBufferSize = INTEROP.execute(cusparseZgemvi_bufferSizeFunction, handle, transA.ordinal(), rows, cols, nnz, bufferSize.getAddress());
                    CUSPARSERegistry.checkCUSPARSEReturnCode(resultBufferSize, cusparseZgemvi_bufferSizeFunction.toString());
                    break;
                }
            }


            long numElements;

            if (bufferSize.getValue() == 0) {
                numElements = 1;
            } else {
                numElements = (long) bufferSize.getValue();
            }

            DeviceArray buffer = new DeviceArray(alpha.getGrCUDAExecutionContext(), numElements, alpha.getElementType());

            // FIXME: getting the runtime from an argument is not very clean, the proxy should maybe hold a direct reference of the runtime;
            alpha.getGrCUDAExecutionContext().getCudaRuntime().cudaDeviceSynchronize();

            args[0] = transA.ordinal();
            args[1] = rows;
            args[2] = cols;
            args[3] = alpha;
            args[4] = matA;
            args[5] = lda;
            args[6] = nnz;
            args[7] = x;
            args[8] = xInd;
            args[9] = beta;
            args[10] = outVec;
            args[11] = idxBase.ordinal();
            args[12] = buffer;

            return args;
        }
    }
}
