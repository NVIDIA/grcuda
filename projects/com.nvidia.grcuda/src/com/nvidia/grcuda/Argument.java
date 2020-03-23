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

import com.oracle.truffle.api.CompilerAsserts;

public class Argument {

    public enum Direction {
        BY_VALUE,
        IN,
        OUT,
        INOUT,
    }

    private int position;
    private final String name;
    private final Type type;
    private final Direction direction;

    /**
     * Create new argument from its components.
     *
     * @param position zero-based position from the left of the argument list
     * @param name argument name
     * @param type data type of the argument
     * @param direction direction of pointer argument (non-pointer arguments are by value)
     */
    Argument(int position, String name, Type type, Direction direction) {
        assert type != Type.VOID : "argument " + name + " has type void";
        assert type != Type.POINTER || direction != Direction.BY_VALUE : "pointer of " + name + " has no direction";
        this.position = position;
        this.name = name;
        this.type = type;
        this.direction = direction;
    }

    /**
     * Create new argument from its components (with position 0).
     *
     * @param name argument name
     * @param type data type of the argument
     * @param direction direction of pointer argument (non-pointer arguments are by value)
     */
    public static Argument createArgument(String name, Type type, Direction direction) {
        return new Argument(0, name, type, direction);
    }

    /**
     * Parse argument string in new argument syntax.
     *
     * argStr: ::= argumentName ":" [direction] typeName
     *
     * direction ::= "in" | "out" | "inout" but only if typeName == "pointer"
     *
     * @param position zero-based position of the argument in the argument list
     * @param argStr the string to be parsed
     * @throws TypeException if `argStr` string cannot be parsed successfully
     * @return Argument
     */
    private static Argument parseNIDLArgumentString(int position, String argStr) throws TypeException {
        String[] nameAndType = argStr.trim().split(":");
        if (nameAndType.length == 1) {
            return parseLegacyArgumentString(position, argStr);
        }
        if (nameAndType.length != 2) {
            throw new IllegalArgumentException("expected argument as \"name: type\", got " + argStr);
        }
        String name = nameAndType[0].trim();
        String extTypeStr = nameAndType[1].trim();
        String[] optDirAndType = extTypeStr.split("(\\s)+");
        if (optDirAndType.length < 1 || optDirAndType.length > 2) {
            throw new IllegalArgumentException("expected type, got " + extTypeStr);
        }
        Direction direction = Direction.BY_VALUE;
        String typeStr = optDirAndType[0];
        if (optDirAndType.length == 2) {
            switch (optDirAndType[0].trim()) {
                case "in":
                    direction = Direction.IN;
                    break;
                case "inout":
                    direction = Direction.INOUT;
                    break;
                case "out":
                    direction = Direction.OUT;
                    break;
                default:
                    throw new IllegalArgumentException("invalid direction: " + optDirAndType[0]);
            }
            typeStr = optDirAndType[1];
        }
        Type type = Type.fromNIDLTypeString(typeStr);
        assertNonVoidType(type, position, argStr);
        return new Argument(position, name, type, direction);
    }

    /**
     * Parse argument string in legacy NFI argument syntax.
     *
     * @param position zero-based position of the argument in the argument list
     * @param argStr the string to be parsed
     * @throws TypeException of the specified type cannot be parsed
     * @return Argument in which the names are "arg1", "arg2", ...
     */
    private static Argument parseLegacyArgumentString(int position, String argStr) throws TypeException {
        String name = "arg" + (position + 1);
        Type type = Type.fromNIDLTypeString(argStr.trim());
        assertNonVoidType(type, position, argStr);
        Direction direction = type == Type.POINTER ? Direction.INOUT : Direction.BY_VALUE;
        return new Argument(position, name, type, direction);
    }

    private static void assertNonVoidType(Type type, int position, String argStr) throws TypeException {
        if (type == Type.VOID) {
            throw new TypeException("argument " + (position + 1) + " has type void in " + argStr);
        }
    }

    public static Argument[] parseSignature(String kernelSignature) throws TypeException {
        CompilerAsserts.neverPartOfCompilation();
        ArrayList<Argument> args = new ArrayList<>();
        for (String s : kernelSignature.trim().split(",")) {
            args.add(parseNIDLArgumentString(args.size(), s.trim()));
        }
        Argument[] argArray = new Argument[args.size()];
        args.toArray(argArray);
        return argArray;
    }

    public Type getType() {
        return type;
    }

    public String getName() {
        return name;
    }

    public Direction getDirection() {
        return direction;
    }

    public void setPosition(int position) {
        this.position = position;
    }

    public int getPosition() {
        return position;
    }

    public String toSignature() {
        return name + ": " + (type == Type.POINTER ? (direction.toString().toLowerCase() + " ") : "") + type.toString().toLowerCase();
    }

    @Override
    public String toString() {
        return "Argument(position=" + position + ", name=" + name + ", type=" + type + ", direction=" + direction + ")";
    }
}
