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

import java.util.ArrayList;

import com.nvidia.grcuda.gpu.computation.GrCUDAComputationalElement;
import com.oracle.truffle.api.CompilerAsserts;

/**
 * Defines a {@link GrCUDAComputationalElement} argument representing the elements of a NIDL/NFI signature.
 * For each argument, store its type, if it's a pointer or a value,
 * and if it's constant (i.e. its content cannot be modified in the computation);
 */
public class Parameter {

    public enum Kind {
        BY_VALUE,
        POINTER_IN,
        POINTER_OUT,
        POINTER_INOUT,
    }

    private int position;
    private final String name;
    private final Type type;
    private final Kind kind;

    protected final boolean isArray;
    protected final boolean isConst;

    /**
     * Create new parameter from its components.
     *
     * @param position zero-based position from the left of the parameter list
     * @param name parameter name
     * @param type data type of the parameter
     * @param kind kind of the parameter (by-value, pointer with direction)
     */
    Parameter(int position, String name, Type type, Kind kind) {
        this.position = position;
        this.name = name;
        this.type = type;
        this.kind = kind;
        this.isArray = kind.equals(Kind.POINTER_IN) || kind.equals(Kind.POINTER_INOUT) || kind.equals(Kind.POINTER_OUT);
        this.isConst = kind.equals(Kind.POINTER_IN) || kind.equals(Kind.BY_VALUE);
    }

    /**
     * Create new pointer parameter from its components (with position 0).
     *
     * @param name parameter name
     * @param type data type of the value to which the pointer points
     * @param kind direction of pointer parameter (allowed values `POINTER_IN`, `POINTER_OUT` and
     *            `POINTER_INOUT`, must not by `BY_VALUE`)
     */
    public static Parameter createPointerParameter(String name, Type type, Kind kind) {
        assert kind != Kind.BY_VALUE : "pointer parameter cannot be by-value";
        return new Parameter(0, name, type, kind);
    }

    /**
     * Create new by-value parameter from its components (with position 0).
     *
     * @param name parameter name
     * @param type data type of the parameter
     */
    public static Parameter createByValueParameter(String name, Type type) {
        return new Parameter(0, name, type, Kind.BY_VALUE);
    }

    /**
     * Parse parameter string in NIDL or legacy Truffle NFI syntax.
     *
     * <pre>
     * NIDL
     *  paramStr ::= parameterName ":" [direction "pointer"] NIDLTypeName
     *  direction ::= "in" | "out" | "inout"
     *
     * NFI
     *  paramStr :: = NFITypeName
     * </pre>
     *
     * @param position zero-based position of the parameter in the parameter list
     * @param param string to be parsed in NIDL or legacy NFI syntax
     * @throws TypeException if {@code param} string cannot be parsed successfully
     * @return Parameter
     */
    private static Parameter parseNIDLOrLegacyParameterString(int position, String param) throws TypeException {
        String paramStr = param.trim();
        if (paramStr.indexOf(':') == -1) {
            // no colon found -> attempt parsing it as a legacy NFI signature
            return parseLegacyParameterString(position, paramStr);
        }
        String[] nameAndType = paramStr.split(":");
        if (nameAndType.length != 2) {
            throw new TypeException("expected parameter as \"name: type\", got " + paramStr);
        }
        String name = nameAndType[0].trim();
        String extTypeStr = nameAndType[1].trim();
        String[] dirPointerAndType = extTypeStr.split("(\\s)+");
        if (dirPointerAndType.length != 1 && dirPointerAndType.length != 3) {
            throw new TypeException("expected type, got " + extTypeStr);
        }
        if (dirPointerAndType.length == 1) {
            Type type = Type.fromNIDLTypeString(dirPointerAndType[0]);
            if (type == Type.NFI_POINTER) {
                // the NFI pointer is not a legal by-value parameter type
                throw new TypeException("invalid type \"pointer\" of by-value parameter");
            }
            if (type == Type.VOID) {
                // the void is not a legal by-value parameter type
                throw new TypeException("invalid type \"pointer\" of by-value parameter");
            }
            return createByValueParameter(name, type);
        } else {
            if (dirPointerAndType[1].equals("pointer")) {
                Type type = Type.fromNIDLTypeString(dirPointerAndType[2]);
                if (type == Type.NFI_POINTER) {
                    // the NFI pointer may not appear as this NIDL pointer's type
                    throw new TypeException("invalid type \"pointer\"");
                }
                switch (dirPointerAndType[0]) {
                    case "in":
                        return createPointerParameter(name, type, Kind.POINTER_IN);
                    case "inout":
                        return createPointerParameter(name, type, Kind.POINTER_INOUT);
                    case "out":
                        return createPointerParameter(name, type, Kind.POINTER_OUT);
                    default:
                        throw new TypeException("invalid direction: " + dirPointerAndType[0] + ", expected \"in\", \"inout\", or \"out\"");
                }
            } else {
                throw new TypeException("expected keyword \"pointer\"");
            }
        }
    }

    /**
     * Parse parameter string in legacy NFI syntax.
     *
     * @param position zero-based position of the parameter in the parameter list
     * @param param the string to be parsed
     * @throws TypeException of the specified type cannot be parsed
     * @return PArameter in which the names are "param1", "param2", ...
     */
    private static Parameter parseLegacyParameterString(int position, String param) throws TypeException {
        String name = "param" + (position + 1);
        // TODO: undesrtand if a pointer is const
        boolean isConst = false;
        Type type = Type.fromNIDLTypeString(param.trim());
        assertNonVoidType(type, position, param);
        Kind kind = type == Type.NFI_POINTER ? Kind.POINTER_INOUT : Kind.BY_VALUE;
        if (isConst && type == Type.NFI_POINTER) {
            kind = Kind.POINTER_IN;
        }
        return new Parameter(position, name, type, kind);
    }

    private static void assertNonVoidType(Type type, int position, String paramStr) throws TypeException {
        if (type == Type.VOID) {
            throw new TypeException("parameter " + (position + 1) + " has type void in " + paramStr);
        }
    }

    public static ArrayList<Parameter> parseParameterSignature(String parameterSignature) throws TypeException {
        CompilerAsserts.neverPartOfCompilation();
        ArrayList<Parameter> params = new ArrayList<>();
        for (String s : parameterSignature.trim().split(",")) {
            params.add(parseNIDLOrLegacyParameterString(params.size(), s.trim()));
        }
        return params;
    }

    public Type getType() {
        return type;
    }

    public boolean isPointer() {
        return !(kind == Kind.BY_VALUE);
    }

    public String getName() {
        return name;
    }

    public Kind getKind() {
        return kind;
    }

    public void setPosition(int position) {
        this.position = position;
    }

    public int getPosition() {
        return position;
    }

    public boolean isArray() {
        return isArray;
    }

    public boolean isConst() {
        return isConst;
    }

    public String getMangledType() {
        // Simple substitution rule for GCC
        // Note that this does not implement the substitution rules
        StringBuffer buf = new StringBuffer();
        switch (kind) {
            case POINTER_IN:
                buf.append("PK");
                break;
            case POINTER_INOUT:
            case POINTER_OUT:
                buf.append("P");
                break;
            default:
        }
        buf.append(type.getMangled());
        return buf.toString();
    }

    public String toNIDLSignatureElement() {
        String pointerStr;
        switch (kind) {
            case POINTER_IN:
                pointerStr = "in pointer ";
                break;
            case POINTER_INOUT:
                pointerStr = "inout pointer ";
                break;
            case POINTER_OUT:
                pointerStr = "out pointer ";
                break;
            default:
                pointerStr = "";
        }
        return name + ": " + pointerStr + type.toString().toLowerCase();
    }

    public String toNFISignatureElement() {
        switch (kind) {
            case POINTER_IN:
            case POINTER_INOUT:
            case POINTER_OUT:
                return "pointer";
            default:
                return type.getNFITypeName();
        }
    }

    public boolean isSynonymousWithPointerTo(Type elementType) {
        if (isPointer()) {
            if (type == Type.NFI_POINTER) {
                // NFI pointers are synonymous to everything
                return true;
            }
            if (type == Type.VOID || elementType == Type.VOID) {
                return true;
            } else {
                return type.isSynonymousWith(elementType);
            }
        } else {
            // value type
            return type.isSynonymousWith(elementType);
        }
    }

    @Override
    public String toString() {
        return "Parameter(position=" + position + ", name=" + name + ", type=" + type + ", kind=" + kind + ")";
    }
}
