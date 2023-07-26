/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.nvidia.grcuda.functions;

import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public abstract class Function implements TruffleObject {

    public static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    private final String name;

    protected Function(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    protected static String expectString(Object argument, String errorMessage) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        try {
            return INTEROP.asString(argument);
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{argument}, errorMessage);
        }
    }

    public static int expectInt(Object number) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        try {
            return INTEROP.asInt(number);
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{number}, "expected integer number argument");
        }
    }

    protected static long expectLong(Object number, String message) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        try {
            return INTEROP.asLong(number);
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{number}, message);
        }
    }

    public static long expectLong(Object number) throws UnsupportedTypeException {
        return expectLong(number, "expected long number argument");
    }

    protected static int expectPositiveInt(Object number) throws UnsupportedTypeException {
        int value = expectInt(number);
        if (value < 0) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{number}, "expected positive int number argument");
        }
        return value;
    }

    public static long expectPositiveLong(Object number) throws UnsupportedTypeException {
        long value = expectLong(number);
        if (value < 0) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{number}, "expected positive long number argument");
        }
        return value;
    }

    public static void checkArgumentLength(Object[] arguments, int expected) throws ArityException {
        if (arguments.length != expected) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(expected, expected, arguments.length);
        }
    }

    public static void checkArgumentLength(Object[] arguments, int minExpected, int maxExpected) throws ArityException {
        if (arguments.length < minExpected || arguments.length > maxExpected) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(minExpected, maxExpected, arguments.length);
        }
    }

    // InteropLibrary implementation

    @ExportMessage
    @SuppressWarnings("static-method")
    public final boolean isExecutable() {
        return true;
    }

    @ExportMessage
    public Object execute(Object[] arguments) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        return call(arguments);
    }

    @SuppressWarnings("unused")
    protected Object call(Object[] arguments) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        throw UnsupportedMessageException.create();
    }
}
