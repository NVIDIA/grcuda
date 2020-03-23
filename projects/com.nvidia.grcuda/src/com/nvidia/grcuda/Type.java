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
package com.nvidia.grcuda;

import com.oracle.truffle.api.CompilerDirectives;

public enum Type {
    BYTE(1, true),
    CHAR(2, true),
    SHORT(2, true),
    INT(4, true),
    LONG(8, true),
    FLOAT(4, true),
    DOUBLE(8, true),
    POINTER(8, false),
    VOID(0, false);

    private final int sizeBytes;
    private final boolean isElementType;

    Type(int sizeBytes, boolean isElementType) {
        this.sizeBytes = sizeBytes;
        this.isElementType = isElementType;
    }

    public int getSizeBytes() {
        return this.sizeBytes;
    }

    public boolean isElementType() {
        return this.isElementType;
    }

    public static Type fromGrCUDATypeString(String type) throws TypeException {
        switch (type) {
            case "char":
                return Type.BYTE;
            case "short":
                return Type.SHORT;
            case "int":
                return Type.INT;
            case "long":
                return Type.LONG;
            case "float":
                return Type.FLOAT;
            case "double":
                return Type.DOUBLE;
            default:
                CompilerDirectives.transferToInterpreter();
                throw new TypeException("invalid type '" + type + "'");
        }
    }

    public static Type fromNIDLTypeString(String type) throws TypeException {
        switch (type) {
            case "pointer":
                return Type.POINTER;
            case "uint64":
            case "sint64":
                return Type.LONG;
            case "uint32":
            case "sint32":
                return Type.INT;
            case "uint16":
            case "sint16":
                return Type.SHORT;
            case "uint8":
            case "sint8":
                return Type.BYTE;
            case "float":
                return Type.FLOAT;
            case "double":
                return Type.DOUBLE;
            case "void":
                return Type.VOID;
            default:
                CompilerDirectives.transferToInterpreter();
                throw new TypeException("invalid type '" + type + "'");
        }
    }

}
