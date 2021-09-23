/*
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
package com.nvidia.grcuda.runtime.computation;

import com.nvidia.grcuda.Type;

import java.util.Objects;

/**
 * Defines a {@link GrCUDAComputationalElement} argument representing the elements of a NFI signature.
 * For each argument, store its type, if it's a pointer,
 * and if it's constant (i.e. its content cannot be modified in the computation).
 * This class also holds a reference to the actual object associated to the argument;
 */
public class ComputationArgumentWithValue extends ComputationArgument {
    private final Object argumentValue;

    public ComputationArgumentWithValue(String name, Type type, Kind kind, Object argumentValue) {
        super(name, type, kind);
        this.argumentValue = argumentValue;
    }

    public ComputationArgumentWithValue(ComputationArgument computationArgument, Object argumentValue) {
        super(computationArgument.getPosition(), computationArgument.getName(), computationArgument.getType(), computationArgument.getKind());
        this.argumentValue = argumentValue;
    }

    public Object getArgumentValue() { return this.argumentValue; }

    @Override
    public String toString() {
        return "ComputationArgumentWithValue(" +
                "argumentValue=" + argumentValue +
                ", type=" + type +
                ", isArray=" + isArray +
                ", isConst=" + isConst +
                ')';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ComputationArgumentWithValue that = (ComputationArgumentWithValue) o;
        return Objects.equals(argumentValue, that.argumentValue);
    }

    @Override
    public int hashCode() {
        return Objects.hash(argumentValue);
    }
}