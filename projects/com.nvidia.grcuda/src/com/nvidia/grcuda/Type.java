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

public enum Type {          // C++ Type in LP64:
    BOOLEAN(1, true),       // bool
    CHAR(1, true),          // char
    SINT8(1, true),         // (signed) char
    UINT8(1, true),         // unsigned char
    CHAR8(1, true),         // char8_t
    CHAR16(2, true),        // char16_t
    SINT16(2, true),        // (signed) short
    UINT16(2, true),        // (unsigned) short
    CHAR32(4, true),        // char32_t
    SINT32(4, true),        // (signed) int
    UINT32(4, true),        // unsigned int
    WCHAR(4, true),         // wchar_t (non Windows)
    SINT64(8, true),        // (signed) long
    UINT64(8, true),        // (unsigned) long
    SLL64(8, true),         // (signed) long long
    ULL64(8, true),         // unsigned long long
    FLOAT(4, true),         // float
    DOUBLE(8, true),        // double
    NFI_POINTER(8, false),  // void* (w/o type) as used in NFI
    STRING(8, false),       // const char*
    VOID(0, false);         // void

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

    public boolean isSynonymousWith(Type type) {
        if (this == type) {
            return true;
        }
        switch (this) {
            case BOOLEAN:
            case CHAR8:
            case UINT8:
                return type == BOOLEAN || type == UINT8 || type == CHAR8;
            case CHAR:
            case SINT8:
                return type == CHAR || type == SINT8;
            case CHAR16:
            case UINT16:
                return type == CHAR16 || type == UINT16;
            case CHAR32:
            case UINT32:
                return type == CHAR32 || type == UINT32;
            case SINT32:
            case WCHAR:
                return type == SINT32 || type == WCHAR;
            case SINT64:
            case SLL64:
                return type == SINT64 || type == SLL64;
            case UINT64:
            case ULL64:
                return type == UINT64 || type == ULL64;
            default:
                return false;
        }
    }

    public static Type fromGrCUDATypeString(String type) throws TypeException {
        switch (type) {
            case "char":
                return Type.CHAR;
            case "short":
                return Type.SINT16;
            case "int":
                return Type.SINT32;
            case "long":
                return Type.SINT64;
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
            case "bool":
                return Type.BOOLEAN;
            case "char":
                return Type.CHAR;
            case "sint8":
                return Type.SINT8;
            case "uint8":
                return Type.UINT8;
            case "char16":
                return Type.CHAR16;
            case "sint16":
                return Type.SINT16;
            case "uint16":
                return Type.UINT16;
            case "char32":
                return Type.CHAR32;
            case "sint32":
                return Type.SINT32;
            case "uint32":
                return Type.UINT32;
            case "wchar":
                return Type.WCHAR;
            case "sint64":
                return Type.SINT64;
            case "uint64":
                return Type.UINT64;
            case "sll64":
                return Type.SLL64;
            case "ull64":
                return Type.ULL64;
            case "float":
                return Type.FLOAT;
            case "double":
                return Type.DOUBLE;
            case "pointer":
                return Type.NFI_POINTER;
            case "string":
                return Type.STRING;
            case "void":
                return Type.VOID;
            default:
                CompilerDirectives.transferToInterpreter();
                throw new TypeException("invalid type '" + type + "'");
        }
    }

    String getMangled() throws TypeException {
        switch (this) {
            case BOOLEAN:
                return "b";
            case CHAR:
                return "c";
            case SINT8:
                return "a";
            case UINT8:
                return "h";
            case CHAR8:
                return "Du";
            case CHAR16:
                return "Ds";
            case SINT16:
                return "s";
            case UINT16:
                return "t";
            case CHAR32:
                return "Di";
            case SINT32:
                return "i";
            case UINT32:
                return "j";
            case WCHAR:
                return "w";
            case SINT64:
                return "l";
            case UINT64:
                return "m";
            case SLL64:
                return "x";
            case ULL64:
                return "y";
            case FLOAT:
                return "f";
            case DOUBLE:
                return "d";
            case VOID:
                return "v";
            default:
                CompilerDirectives.transferToInterpreter();
                throw new TypeException("no mangling character for type '" + this + "'");
        }
    }
}
