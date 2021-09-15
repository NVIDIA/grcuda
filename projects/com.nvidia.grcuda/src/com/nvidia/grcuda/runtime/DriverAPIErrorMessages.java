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
package com.nvidia.grcuda.runtime;

import java.util.EnumSet;
import java.util.HashMap;

public final class DriverAPIErrorMessages {

    private static final HashMap<Integer, String> codeMap = new HashMap<>();

    static {
        EnumSet.allOf(ErrorCode.class).forEach(code -> codeMap.put(code.getErrorCode(), code.name()));
    }

    public static String getString(int errorCode) {
        String str = codeMap.get(errorCode);
        if (str == null) {
            return "unknown error code (" + errorCode + ")";
        }
        return str;
    }

    /**
     * Error codes from cuda.h of the CUDA Runtime API.
     */
    private enum ErrorCode {
        CUDA_SUCCESS(0),
        CUDA_ERROR_INVALID_VALUE(1),
        CUDA_ERROR_OUT_OF_MEMORY(2),
        CUDA_ERROR_NOT_INITIALIZED(3),
        CUDA_ERROR_DEINITIALIZED(4),
        CUDA_ERROR_PROFILER_DISABLED(5),
        CUDA_ERROR_PROFILER_NOT_INITIALIZED(6),
        CUDA_ERROR_PROFILER_ALREADY_STARTED(7),
        CUDA_ERROR_PROFILER_ALREADY_STOPPED(8),
        CUDA_ERROR_NO_DEVICE(100),
        CUDA_ERROR_INVALID_DEVICE(101),
        CUDA_ERROR_INVALID_IMAGE(200),
        CUDA_ERROR_INVALID_CONTEXT(201),
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT(202),
        CUDA_ERROR_MAP_FAILED(205),
        CUDA_ERROR_UNMAP_FAILED(206),
        CUDA_ERROR_ARRAY_IS_MAPPED(207),
        CUDA_ERROR_ALREADY_MAPPED(208),
        CUDA_ERROR_NO_BINARY_FOR_GPU(209),
        CUDA_ERROR_ALREADY_ACQUIRED(210),
        CUDA_ERROR_NOT_MAPPED(211),
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY(212),
        CUDA_ERROR_NOT_MAPPED_AS_POINTER(213),
        CUDA_ERROR_ECC_UNCORRECTABLE(214),
        CUDA_ERROR_UNSUPPORTED_LIMIT(215),
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE(216),
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED(217),
        CUDA_ERROR_INVALID_PTX(218),
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT(219),
        CUDA_ERROR_NVLINK_UNCORRECTABLE(220),
        CUDA_ERROR_JIT_COMPILER_NOT_FOUND(221),
        CUDA_ERROR_INVALID_SOURCE(300),
        CUDA_ERROR_FILE_NOT_FOUND(301),
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND(302),
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED(303),
        CUDA_ERROR_OPERATING_SYSTEM(304),
        CUDA_ERROR_INVALID_HANDLE(400),
        CUDA_ERROR_NOT_FOUND(500),
        CUDA_ERROR_NOT_READY(600),
        CUDA_ERROR_ILLEGAL_ADDRESS(700),
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES(701),
        CUDA_ERROR_LAUNCH_TIMEOUT(702),
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING(703),
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED(704),
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED(705),
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE(708),
        CUDA_ERROR_CONTEXT_IS_DESTROYED(709),
        CUDA_ERROR_ASSERT(710),
        CUDA_ERROR_TOO_MANY_PEERS(711),
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED(712),
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED(713),
        CUDA_ERROR_HARDWARE_STACK_ERROR(714),
        CUDA_ERROR_ILLEGAL_INSTRUCTION(715),
        CUDA_ERROR_MISALIGNED_ADDRESS(716),
        CUDA_ERROR_INVALID_ADDRESS_SPACE(717),
        CUDA_ERROR_INVALID_PC(718),
        CUDA_ERROR_LAUNCH_FAILED(719),
        CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE(720),
        CUDA_ERROR_NOT_PERMITTED(800),
        CUDA_ERROR_NOT_SUPPORTED(801),
        CUDA_ERROR_UNKNOWN(999);

        private final int errorCode;

        ErrorCode(int errorCode) {
            this.errorCode = errorCode;
        }

        public int getErrorCode() {
            return errorCode;
        }

        @Override
        public String toString() {
            return name() + "(" + errorCode + ")";
        }
    }
}
