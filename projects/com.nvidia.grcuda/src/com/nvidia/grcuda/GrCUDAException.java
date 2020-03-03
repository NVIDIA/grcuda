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
package com.nvidia.grcuda;

import java.util.Arrays;
import java.util.Optional;

import com.oracle.truffle.api.TruffleException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.nodes.Node;

public final class GrCUDAException extends RuntimeException implements TruffleException {
    private static final long serialVersionUID = 8614211550329856579L;

    private final Node node;

    public GrCUDAException(String message) {
        this(message, null);
    }

    public GrCUDAException(String message, Node node) {
        super(message);
        this.node = node;
    }

    public GrCUDAException(InteropException e) {
        this(e.getMessage());
    }

    public GrCUDAException(int errorCode, String[] functionName) {
        this("CUDA error " + errorCode + " in " + format(functionName));
    }

    public GrCUDAException(int errorCode, String message, String[] functionName) {
        this(message + '(' + errorCode + ") in " + format(functionName));
    }

    public static String format(String... name) {
        Optional<String> result = Arrays.asList(name).stream().reduce((a, b) -> a + "::" + b);
        return result.orElse("<empty>");
    }

    @Override
    public Node getLocation() {
        return node;
    }
}
