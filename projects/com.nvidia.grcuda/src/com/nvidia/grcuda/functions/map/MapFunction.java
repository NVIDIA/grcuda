/*
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
package com.nvidia.grcuda.functions.map;

import java.util.Arrays;

import com.nvidia.grcuda.DeviceArray.MemberSet;
import com.nvidia.grcuda.functions.Function;
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
public final class MapFunction extends Function {

    protected static final String RET = "ret";
    protected static final String ARG = "arg";
    protected static final String SIZE = "size";
    protected static final String VALUE = "value";
    private static final MemberSet MEMBERS = new MemberSet(ARG, SIZE, VALUE);

    private final Object returnValue;

    public MapFunction() {
        super("map", "");
        this.returnValue = new MapArgObject(new MapArgObjectValue("return", null, new Object[0]));
    }

    public MapFunction(Object returnValue) {
        super("map", "");
        this.returnValue = returnValue;
    }

    // Java API

    @SuppressWarnings("static-method")
    public MapFunction ret(Object value) throws ArityException {
        return new MapRetFunction().execute(new Object[]{value});
    }

    @SuppressWarnings("static-method")
    public MapArgObject arg(String name) throws ArityException, UnsupportedTypeException {
        return new MapArgFunction().execute(new Object[]{name});
    }

    @SuppressWarnings("static-method")
    public MapArgObject size(Object... arguments) throws ArityException, UnsupportedTypeException {
        return new MapSizeFunction().execute(arguments);
    }

    @SuppressWarnings("static-method")
    public MapArgObject value(String name) throws ArityException, UnsupportedTypeException {
        return new MapValueFunction().execute(new Object[]{name});
    }

    @SuppressWarnings("static-method")
    public MapArgObject value(String name, Object function, Object... arguments) throws ArityException, UnsupportedTypeException {
        Object[] args = new Object[arguments.length + 2];
        args[0] = name;
        args[1] = function;
        System.arraycopy(arguments, 0, args, 2, arguments.length);
        return new MapValueFunction().execute(args);
    }

    public MappedFunction map(Object function, Object... arguments) throws ArityException, UnsupportedTypeException {
        Object[] args = new Object[arguments.length + 1];
        args[0] = function;
        System.arraycopy(arguments, 0, args, 1, arguments.length);
        return execute(args);
    }

    // Interop API

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
            return new MapRetFunction();
        }

        @Specialization(guards = "ARG.equals(member)")
        static MapFunctionBase readMemberArg(MapFunction receiver, String member) {
            return new MapArgFunction();
        }

        @Specialization(guards = "SIZE.equals(member)")
        static MapFunctionBase readMemberSize(MapFunction receiver, String member) {
            return new MapSizeFunction();
        }

        @Specialization(guards = "VALUE.equals(member)")
        static MapFunctionBase readMemberValue(MapFunction receiver, String member) {
            return new MapValueFunction();
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
                    @CachedLibrary(limit = "1") InteropLibrary interopRead,
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
                    throw new RuntimeException("mismatch between isString and asString");
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
            throw new RuntimeException("unable to bind argument " + argument);
        }
    }
}

abstract class MapFunctionBase implements TruffleObject {

    protected static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    protected void checkArity(Object[] arguments, int expectedArity) throws ArityException {
        if (arguments.length != expectedArity) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(expectedArity, arguments.length);
        }
    }

    protected String asString(Object argument, String message) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        if (INTEROP.isString(argument)) {
            try {
                return INTEROP.asString(argument);
            } catch (UnsupportedMessageException e) {
                // fallthrough
            }
        }
        throw UnsupportedTypeException.create(new Object[]{argument}, message);
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapArgFunction extends MapFunctionBase {

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    MapArgObject execute(Object[] arguments) throws ArityException, UnsupportedTypeException {
        checkArity(arguments, 1);
        String name = asString(arguments[0], "name of input argument expected");
        return new MapArgObject(new MapArgObjectArgument(name));
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapRetFunction extends MapFunctionBase {

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    MapFunction execute(Object[] arguments) throws ArityException {
        checkArity(arguments, 1);
        return new MapFunction(arguments[0]);
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapSizeFunction extends MapFunctionBase {

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    @TruffleBoundary
    @SuppressWarnings("static-method")
    MapArgObject execute(Object[] arguments) throws ArityException, UnsupportedTypeException {
        if (arguments.length == 0) {
            throw ArityException.create(1, 0);
        }
        MapArgObjectBase[] values = new MapArgObjectBase[arguments.length];
        for (int i = 0; i < arguments.length; i++) {
            Object argument = arguments[i];
            if (argument instanceof MapArgObject) {
                MapArgObject object = (MapArgObject) argument;
                values[i] = object.value;
            } else {
                throw UnsupportedTypeException.create(new Object[]{argument}, "expected argument object");
            }
        }
        return new MapArgObject(new MapArgObjectSize(values));
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapValueFunction extends MapFunctionBase {

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    MapArgObject execute(Object[] arguments) throws ArityException, UnsupportedTypeException {
        if (arguments.length < 1) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(1, arguments.length);
        }
        String name = asString(arguments[0], "name of created value expected");
        if (arguments.length == 1) {
            return new MapArgObject(new MapArgObjectValue(name, null, new Object[0]));
        } else {
            Object function = arguments[1];
            Object[] args = Arrays.copyOfRange(arguments, 2, arguments.length);
            return new MapArgObject(new MapArgObjectValue(name, function, args));
        }
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapReturnFunction extends MapFunctionBase {

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    MapArgObject execute(Object[] arguments) throws ArityException, UnsupportedTypeException {
        if (arguments.length < 1) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(1, arguments.length);
        }
        String name = asString(arguments[0], "name of created value expected");
        if (arguments.length == 1) {
            return new MapArgObject(new MapArgObjectValue(name, null, new Object[0]));
        } else {
            Object function = arguments[1];
            Object[] args = Arrays.copyOfRange(arguments, 2, arguments.length);
            return new MapArgObject(new MapArgObjectValue(name, function, args));
        }
    }
}
