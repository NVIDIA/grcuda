/*
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
package com.nvidia.grcuda.functions.map;

import java.util.Arrays;

import com.nvidia.grcuda.DeviceArray.MemberSet;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.dsl.Fallback;
import com.oracle.truffle.api.dsl.Specialization;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public final class MapFunction extends com.nvidia.grcuda.functions.Function {

    protected static final String RET = "ret";
    protected static final String ARG = "arg";
    protected static final String SIZE = "size";
    protected static final String VALUE = "value";
    private static final MemberSet MEMBERS = new MemberSet(RET, ARG, SIZE, VALUE);

    private final Object returnValue;

    public MapFunction() {
        this(new MapArgObject(new MapArgObjectValue("return", null, new Object[0])));
    }

    public MapFunction(Object returnValue) {
        super("Map");
        this.returnValue = returnValue;
    }

    // Java API

    @SuppressWarnings("static-method")
    public MapFunction ret(Object value) {
        return new MapFunction(value);
    }

    @SuppressWarnings("static-method")
    public MapArgObject arg(String name) {
        return new MapArgObject(new MapArgObjectArgument(name));
    }

    @SuppressWarnings("static-method")
    public MapArgObject size(MapArgObject first, MapArgObject... additional) {
        MapArgObjectBase[] values = new MapArgObjectBase[1 + additional.length];
        values[0] = first.value;
        for (int i = 0; i < additional.length; i++) {
            values[i + 1] = additional[i].value;
        }
        return new MapArgObject(new MapArgObjectSize(values));
    }

    @SuppressWarnings("static-method")
    public MapArgObject value(String name) {
        return new MapArgObject(new MapArgObjectValue(name, null, new Object[0]));
    }

    @SuppressWarnings("static-method")
    public MapArgObject value(String name, Object function, Object... arguments) {
        return new MapArgObject(new MapArgObjectValue(name, function, arguments));

    }

    public MappedFunction map(Object function, Object... arguments) throws ArityException, UnsupportedTypeException {
        Object[] args = new Object[arguments.length + 1];
        args[0] = function;
        System.arraycopy(arguments, 0, args, 1, arguments.length);
        return execute(args);
    }

    // Interop API

    protected static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    static void checkArity(Object[] arguments, int expectedArity) throws ArityException {
        if (arguments.length != expectedArity) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(expectedArity, arguments.length);
        }
    }

    static String checkString(Object argument, String message) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        try {
            return INTEROP.asString(argument);
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{argument}, message);
        }
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return MEMBERS;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String member) {
        return MEMBERS.constainsValue(member);
    }

    @ExportMessage
    @SuppressWarnings("unused")
    abstract static class ReadMember {
        @Specialization(guards = "RET.equals(member)")
        static MapFunctionBase readMemberRet(MapFunction receiver, String member) {
            return new MapFunctionBase(arguments -> {
                checkArity(arguments, 1);
                return receiver.ret(arguments[0]);
            });
        }

        @Specialization(guards = "ARG.equals(member)")
        static MapFunctionBase readMemberArg(MapFunction receiver, String member) {
            return new MapFunctionBase(arguments -> {
                checkArity(arguments, 1);
                String name = checkString(arguments[0], "name of input argument expected");
                return receiver.arg(name);
            });
        }

        @Specialization(guards = "SIZE.equals(member)")
        static MapFunctionBase readMemberSize(MapFunction receiver, String member) {
            return new MapFunctionBase(arguments -> {
                if (arguments.length == 0) {
                    throw ArityException.create(1, 0);
                }
                try {
                    return receiver.size((MapArgObject) arguments[0], Arrays.copyOfRange(arguments, 1, arguments.length, MapArgObject[].class));
                } catch (ClassCastException | ArrayStoreException e) {
                    throw UnsupportedTypeException.create(arguments, "expected argument objects");
                }
            });
        }

        @Specialization(guards = "VALUE.equals(member)")
        static MapFunctionBase readMemberValue(MapFunction receiver, String member) {
            return new MapFunctionBase(arguments -> {
                if (arguments.length < 1) {
                    throw ArityException.create(1, arguments.length);
                }
                String name = checkString(arguments[0], "name of created value expected");
                if (arguments.length == 1) {
                    return receiver.value(name);
                } else {
                    Object function = arguments[1];
                    Object[] args = Arrays.copyOfRange(arguments, 2, arguments.length);
                    return receiver.value(name, function, args);
                }
            });
        }

        @Fallback
        static MapFunctionBase readMemberOther(MapFunction receiver, String member) throws UnknownIdentifierException {
            throw UnknownIdentifierException.create(member);
        }
    }

    @ExportMessage
    boolean isMemberInvocable(String member) {
        return isMemberReadable(member);
    }

    @ExportMessage
    Object invokeMember(String member, Object[] arguments,
                    @CachedLibrary("this") InteropLibrary interopRead,
                    @CachedLibrary(limit = "1") InteropLibrary interopExecute) throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        return interopExecute.execute(interopRead.readMember(this, member), arguments);
    }

    @Override
    @ExportMessage
    @TruffleBoundary
    public MappedFunction execute(Object[] arguments) throws ArityException, UnsupportedTypeException {
        if (arguments.length == 0) {
            throw ArityException.create(1, 0);
        }
        Object function = arguments[0];
        if (!INTEROP.isExecutable(function)) {
            throw UnsupportedTypeException.create(arguments, "expecting executable function as first argument");
        }
        String description = null;
        if (arguments.length > 1) {
            Object last = arguments[arguments.length - 1];
            if (INTEROP.isString(last)) {
                try {
                    description = INTEROP.asString(last);
                } catch (UnsupportedMessageException e) {
                    throw new GrCUDAInternalException("mismatch between isString and asString");
                }
            }
        }
        Object[] values = new Object[arguments.length - 1 - (description == null ? 0 : 1)];
        ArgumentSet argSet = new ArgumentSet();
        ArgumentSet shreddedArgSet = new ArgumentSet();
        ArgumentSet valueSet = new ArgumentSet();

        for (int i = 0; i < values.length; i++) {
            values[i] = bindArgument(arguments[i + 1], argSet, shreddedArgSet, valueSet);
        }
        Object boundReturn = bindArgument(returnValue, argSet, shreddedArgSet, valueSet);
        int[] shreddedIndexes = new int[shreddedArgSet.nameList.size()];
        for (String name : shreddedArgSet.nameList.getKeys()) {
            shreddedIndexes[shreddedArgSet.nameList.get(name)] = argSet.readMember(name);
        }
        Integer returnValueIndex = valueSet.nameList.get("return");
        return new MappedFunction(function, values, shreddedIndexes, valueSet.nameList.size(), boundReturn, returnValueIndex, description);
    }

    private static Object bindArgument(Object argument, ArgumentSet argSet, ArgumentSet shreddedArgSet, ArgumentSet valueSet) throws UnsupportedTypeException {
        try {
            if (INTEROP.isMemberInvocable(argument, "bind")) {
                return INTEROP.invokeMember(argument, "bind", argSet, shreddedArgSet, valueSet);
            } else {
                Object readMember = INTEROP.readMember(argument, "bind");
                return INTEROP.execute(readMember, argSet, shreddedArgSet, valueSet);
            }
        } catch (UnsupportedMessageException | UnknownIdentifierException | ArityException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAInternalException("unable to bind argument " + argument);
        }
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapFunctionBase implements TruffleObject {

    interface Spec {
        Object apply(Object[] arguments) throws UnsupportedTypeException, ArityException, UnsupportedMessageException;
    }

    private final Spec op;

    MapFunctionBase(Spec op) {
        this.op = op;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    @TruffleBoundary
    Object execute(Object[] arguments) throws UnsupportedTypeException, ArityException, UnsupportedMessageException {
        return op.apply(arguments);
    }
}
