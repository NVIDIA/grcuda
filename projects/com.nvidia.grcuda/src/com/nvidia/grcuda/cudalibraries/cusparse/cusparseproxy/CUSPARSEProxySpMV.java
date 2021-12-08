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
import static com.nvidia.grcuda.functions.Function.expectLong;

import static com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.CUSPARSEIndexType;
import static com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.CUSPARSEIndexBase;
import static com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.CUDADataType;
import static com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.CUSPARSESpMVAlg;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import com.sun.jdi.Value;

public class CUSPARSEProxySpMV extends CUSPARSEProxy {

    public enum CUSPARSESpMVMatrixType {
        SPMV_MATRIX_TYPE_COO,
        SPMV_MATRIX_TYPE_CSR
    }

    // Number of arguments expected to directly call the original SpMV function in cuSPARSE;
    private final int NUM_SPMV_ARGS_READ = 9;
    // Number of arguments expected to call SpMV by automatically wrapping input arrays;
    private final int NUM_SPMV_ARGS_WRAPPED = 14;

    public CUSPARSEProxySpMV(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs, long handle) throws UnsupportedTypeException, UnsupportedMessageException, ArityException {
        this.initializeNfi();
        if (rawArgs.length == NUM_SPMV_ARGS_READ) {
            return rawArgs;
        } else {
            args = new Object[NUM_SPMV_ARGS_WRAPPED];

            // v1 and v2 can be X, Y, rowPtr
            DeviceArray v1 = (DeviceArray) rawArgs[5];
            DeviceArray v2 = (DeviceArray) rawArgs[6];
            DeviceArray values = (DeviceArray) rawArgs[7];

            // Last argument is the matrix type
            CUSPARSESpMVMatrixType matrixType = CUSPARSESpMVMatrixType.values()[expectInt(rawArgs[rawArgs.length - 1])];

            UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object matDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();

            CUSPARSERegistry.CUSPARSEOperation opA = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(rawArgs[0])];
            DeviceArray alpha = (DeviceArray) rawArgs[1];
            long rows = expectLong(rawArgs[2]);
            long cols = expectLong(rawArgs[3]);
            long nnz = expectLong(rawArgs[4]);
            CUSPARSEIndexType idxType = CUSPARSEIndexType.values()[expectInt(rawArgs[8])];
            CUSPARSEIndexBase idxBase = CUSPARSEIndexBase.values()[expectInt(rawArgs[9])];
            CUDADataType valueType = CUDADataType.values()[expectInt(rawArgs[10])];
            DeviceArray valuesX = (DeviceArray) rawArgs[11];
            CUDADataType valueTypeVec = CUDADataType.values()[expectInt(rawArgs[12])];
            DeviceArray beta = (DeviceArray) rawArgs[13];
            DeviceArray valuesY = (DeviceArray) rawArgs[14];
            CUSPARSESpMVAlg alg = CUSPARSESpMVAlg.values()[expectInt(rawArgs[15])];

            switch (matrixType){
                case SPMV_MATRIX_TYPE_COO: {
                    Object resultCoo = INTEROP.execute(cusparseCreateCooFunction, matDescr.getAddress(), rows, cols, nnz, v1, v2, values, idxType.ordinal(), idxBase.ordinal(),
                            valueType.ordinal());
                    break;
                }
                case SPMV_MATRIX_TYPE_CSR: {
                    Object resultCsr = INTEROP.execute(cusparseCreateCsrFunction, matDescr.getAddress(), rows, cols, nnz, v1, v2, values, idxType.ordinal(),
                            idxType.ordinal(), idxBase.ordinal(), valueType.ordinal());
                    break;
                }

            }

            // create dense vectors X and Y descriptors
            Object resultX = INTEROP.execute(cusparseCreateDnVecFunction, dnVecXDescr.getAddress(), cols, valuesX, valueTypeVec.ordinal());
            Object resultY = INTEROP.execute(cusparseCreateDnVecFunction, dnVecYDescr.getAddress(), cols, valuesY, valueTypeVec.ordinal());

            // create buffer
            Object resultBufferSize = INTEROP.execute(cusparseSpMV_bufferSizeFunction, handle, opA.ordinal(), alpha, matDescr.getValue(), dnVecXDescr.getValue(), beta,
                                dnVecYDescr.getValue(), valueType.ordinal(), alg.ordinal(), bufferSize.getAddress());

            long numElements;

            if (bufferSize.getValue() == 0) {
                numElements = 1;
            } else {
                numElements = bufferSize.getValue() / 4;
            }

            DeviceArray buffer = new DeviceArray(alpha.getGrCUDAExecutionContext(), numElements, alpha.getElementType());

            // FIXME: getting the runtime from an argument is not very clean, the proxy should maybe hold a direct reference of the runtime;
            alpha.getGrCUDAExecutionContext().getCudaRuntime().cudaDeviceSynchronize();

            // format new arguments
            args[0] = opA.ordinal();
            args[1] = alpha;
            args[2] = matDescr.getValue();
            args[3] = dnVecXDescr.getValue();
            args[4] = beta;
            args[5] = dnVecYDescr.getValue();
            args[6] = valueType.ordinal();
            args[7] = alg.ordinal();
            args[8] = buffer;

            // Extra arguments, required to track dependencies on the original input arrays;
            args[9] = v1;
            args[10] = v2;
            args[11] = values;
            args[12] = valuesX;
            args[13] = valuesY;

            return args;
        }
    }
}
