/*
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
package com.nvidia.grcuda.functions.map;

import static com.nvidia.grcuda.functions.map.MapFunction.checkArity;

import java.util.Arrays;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.CompilationFinal;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Specialization;
import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.nodes.ExplodeLoop;
import com.oracle.truffle.api.profiles.ConditionProfile;
import com.oracle.truffle.api.profiles.PrimitiveValueProfile;

final class MapException extends AbstractTruffleException {

    private static final long serialVersionUID = -1472390370115466332L;

    MapException(String message) {
        super(message);
    }
}

/**
 * Base class for "abstract" argument descriptors.
 */
abstract class MapArgObjectBase implements TruffleObject {
    protected static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    protected abstract MapBoundArgObjectBase bind(Object argumentSet, Object shreddedArgumentSet, Object valueSet)
                    throws UnsupportedMessageException, ArityException, UnsupportedTypeException;
}

/**
 * Base class for "bound" argument descriptors, which have been processed in the context of a
 * specific list of arguments.
 */
abstract class MapBoundArgObjectBase implements TruffleObject {
}

final class MapArgObjectArgument extends MapArgObjectBase {

    final String name;

    MapArgObjectArgument(String name) {
        this.name = name;
    }

    @Override
    protected MapBoundArgObjectBase bind(Object argumentSet, Object shreddedArgumentSet, Object valueSet) {
        int index;
        try {
            index = INTEROP.asInt(INTEROP.readMember(argumentSet, name));
        } catch (UnsupportedMessageException | UnknownIdentifierException e) {
            CompilerDirectives.transferToInterpreter();
            throw new MapException("cannot resolve argument index for '" + name + "'");
        }
        return new MapBoundArgObjectArgument(name, index);
    }

    @Override
    public String toString() {
        return "arg('" + name + "')";
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapBoundArgObjectArgument extends MapBoundArgObjectBase {
    final String name;
    final int index;

    MapBoundArgObjectArgument(String name, int index) {
        this.name = name;
        this.index = index;
    }

    @Override
    public String toString() {
        return "arg('" + name + "', " + index + ")";
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary(limit = "1") InteropLibrary interop) throws ArityException {
        checkArity(arguments, 3);
        try {
            return interop.readArrayElement(arguments[0], index);
        } catch (InvalidArrayIndexException e) {
            CompilerDirectives.transferToInterpreter();
            throw new MapException("cannot get argument '" + name + "' at index " + index + ", too few arguments");
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw new MapException("cannot get argument argument '" + name + "' at index " + index + ", invalid arguments object");
        }
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapBoundShreddedArgObjectArgument extends MapBoundArgObjectBase {
    final String name;
    final int index;

    MapBoundShreddedArgObjectArgument(String name, int index) {
        this.name = name;
        this.index = index;
    }

    @Override
    public String toString() {
        return "arg('" + name + "', " + index + ").shred()";
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary(limit = "1") InteropLibrary interop) throws ArityException {
        checkArity(arguments, 3);
        try {
            return interop.readArrayElement(arguments[1], index);
        } catch (InvalidArrayIndexException e) {
            CompilerDirectives.transferToInterpreter();
            throw new MapException("cannot get shredded argument '" + name + "' at index " + index + ", too few arguments");
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw new MapException("cannot get shredded argument '" + name + "' at index " + index + ", invalid arguments object");
        }
    }
}

final class MapArgObjectSize extends MapArgObjectBase {
    @CompilationFinal(dimensions = 1) final MapArgObjectBase[] values;

    MapArgObjectSize(MapArgObjectBase[] values) {
        assert values.length >= 1;
        this.values = values;
    }

    @Override
    protected MapBoundArgObjectBase bind(Object argumentSet, Object shreddedArgumentSet, Object valueSet)
                    throws UnsupportedMessageException, ArityException, UnsupportedTypeException {
        MapBoundArgObjectBase[] newValues = new MapBoundArgObjectBase[values.length];
        for (int i = 0; i < values.length; i++) {
            newValues[i] = values[i].bind(argumentSet, shreddedArgumentSet, valueSet);
        }
        return new MapBoundArgObjectSize(newValues);
    }

    @Override
    public String toString() {
        return "size" + Arrays.toString(values);
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapBoundArgObjectSize extends MapBoundArgObjectBase {
    @CompilationFinal(dimensions = 1) final MapBoundArgObjectBase[] values;

    MapBoundArgObjectSize(MapBoundArgObjectBase[] values) {
        assert values.length >= 1;
        this.values = values;
    }

    @Override
    public String toString() {
        return "size" + Arrays.toString(values);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    long execute(Object[] arguments,
                    @CachedLibrary(limit = "1") InteropLibrary firstExecInterop,
                    @CachedLibrary(limit = "2") InteropLibrary firstSizeInterop,
                    @CachedLibrary(limit = "2") InteropLibrary execInterop,
                    @CachedLibrary(limit = "2") InteropLibrary sizeInterop,
                    @Cached("createEqualityProfile()") PrimitiveValueProfile lengthProfile) throws ArityException {
        if (arguments.length != 3) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(3, arguments.length);
        }
        long size;
        try {
            size = firstSizeInterop.getArraySize(firstExecInterop.execute(values[0], arguments));
        } catch (UnsupportedTypeException | UnsupportedMessageException | ArityException e) {
            CompilerDirectives.transferToInterpreter();
            throw new MapException("cannot get size from argument " + values[0]);
        }
        int length = lengthProfile.profile(values.length);
        for (int i = 1; i < length; i++) {
            long otherSize;
            try {
                otherSize = sizeInterop.getArraySize(execInterop.execute(values[i], arguments));
            } catch (UnsupportedTypeException | UnsupportedMessageException | ArityException e) {
                CompilerDirectives.transferToInterpreter();
                throw new MapException("cannot get size from argument " + values[i]);
            }
            if (otherSize != size) {
                CompilerDirectives.transferToInterpreter();
                throw new MapException("mismatching size for parameters " + values[0] + " and " + values[i]);
            }
        }
        return size;
    }
}

final class MapArgObjectValue extends MapArgObjectBase {

    private static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    final Object function;
    @CompilationFinal(dimensions = 1) final Object[] args;
    private final String name;

    MapArgObjectValue(String name, Object function, Object[] args) {
        this.name = name;
        this.function = function;
        this.args = args;
    }

    @Override
    protected MapBoundArgObjectBase bind(Object argumentSet, Object shreddedArgumentSet, Object valueSet)
                    throws UnsupportedMessageException, ArityException, UnsupportedTypeException {
        Object[] newArgs = new Object[args.length];
        for (int i = 0; i < args.length; i++) {
            try {
                newArgs[i] = INTEROP.invokeMember(args[i], "bind", argumentSet, shreddedArgumentSet, valueSet);
            } catch (UnknownIdentifierException e) {
                CompilerDirectives.transferToInterpreter();
                throw new MapException("cannot bind parameter " + i + " of value");
            }
        }
        int index;
        try {
            index = INTEROP.asInt(INTEROP.readMember(valueSet, name));
        } catch (UnsupportedMessageException | UnknownIdentifierException e) {
            CompilerDirectives.transferToInterpreter();
            throw new MapException("cannot resolve value index for '" + name + "'");
        }
        return new MapBoundArgObjectValue(name, index, function, newArgs);
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("value('").append(name);
        if (function != null) {
            str.append("', ").append(function.getClass().getSimpleName());
        }
        for (int i = 0; i < args.length; i++) {
            str.append(", ").append(args[i]);
        }
        return str.append(")").toString();
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapBoundArgObjectValue extends MapBoundArgObjectBase {

    final Object function;
    @CompilationFinal(dimensions = 1) final Object[] args;
    private final String name;
    private final int index;

    MapBoundArgObjectValue(String name, int index, Object function, Object[] args) {
        this.name = name;
        this.index = index;
        this.function = function;
        this.args = args;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("value('").append(name).append("', ").append(index);
        if (function != null) {
            str.append("', ").append(function.getClass().getSimpleName());
        }
        for (int i = 0; i < args.length; i++) {
            str.append(", ").append(args[i]);
        }
        return str.append(")").toString();
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary(limit = "1") InteropLibrary firstExecInterop,
                    @CachedLibrary(limit = "2") InteropLibrary execInterop,
                    @CachedLibrary(limit = "2") InteropLibrary functionInterop,
                    @CachedLibrary(limit = "2") InteropLibrary elementInterop,
                    @Cached("createBinaryProfile()") ConditionProfile hasFunctionProfile,
                    @Cached("createEqualityProfile()") PrimitiveValueProfile lengthProfile) throws ArityException {
        if (arguments.length != 3) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(3, arguments.length);
        }
        int length = lengthProfile.profile(args.length);
        Object[] mappedArgs = new Object[length];
        if (length > 0) {
            try {
                mappedArgs[0] = firstExecInterop.execute(args[0], arguments);
            } catch (UnsupportedTypeException | UnsupportedMessageException | ArityException e) {
                CompilerDirectives.transferToInterpreter();
                throw new MapException("cannot get argument " + args[0] + " for " + this);
            }
            for (int i = 1; i < length; i++) {
                try {
                    mappedArgs[i] = execInterop.execute(args[i], arguments);
                } catch (UnsupportedTypeException | UnsupportedMessageException | ArityException e) {
                    CompilerDirectives.transferToInterpreter();
                    throw new MapException("cannot get argument " + args[i] + " for " + this);
                }
            }
        }

        try {
            Object value;
            if (hasFunctionProfile.profile(function != null)) {
                value = functionInterop.execute(function, mappedArgs);
                elementInterop.writeArrayElement(arguments[2], index, value);
            } else {
                value = elementInterop.readArrayElement(arguments[2], index);
                if (value == null) {
                    CompilerDirectives.transferToInterpreter();
                    throw new MapException("value is read before it is produced in " + this);
                }
            }
            return value;
        } catch (UnsupportedTypeException | UnsupportedMessageException | InvalidArrayIndexException e) {
            throw new MapException("cannot execute value function " + this);
        }
    }
}

final class MapArgObjectMember extends MapArgObjectBase {
    final MapArgObjectBase parent;
    final String name;

    MapArgObjectMember(String name, MapArgObjectBase parent) {
        this.parent = parent;
        this.name = name;
    }

    @Override
    protected MapBoundArgObjectBase bind(Object argumentSet, Object shreddedArgumentSet, Object valueSet)
                    throws UnsupportedMessageException, ArityException, UnsupportedTypeException {
        return new MapBoundArgObjectMember(name, parent.bind(argumentSet, shreddedArgumentSet, valueSet));
    }

    @Override
    public String toString() {
        return parent + "." + name;
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapBoundArgObjectMember extends MapBoundArgObjectBase {
    final MapBoundArgObjectBase parent;
    final String name;

    MapBoundArgObjectMember(String name, MapBoundArgObjectBase parent) {
        this.parent = parent;
        this.name = name;
    }

    @Override
    public String toString() {
        return parent + "." + name;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary("this.parent") InteropLibrary parentInterop,
                    @CachedLibrary(limit = "2") InteropLibrary memberInterop) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        if (arguments.length != 3) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(3, arguments.length);
        }
        Object value = parentInterop.execute(parent, arguments);
        try {
            return memberInterop.readMember(value, name);
        } catch (UnsupportedMessageException | UnknownIdentifierException e) {
            CompilerDirectives.transferToInterpreter();
            throw new MapException("cannot read member '" + name + "' from argument " + parent);
        }
    }
}

final class MapArgObjectElement extends MapArgObjectBase {
    final MapArgObjectBase parent;
    final long index;

    MapArgObjectElement(long index, MapArgObjectBase parent) {
        this.parent = parent;
        this.index = index;
    }

    @Override
    protected MapBoundArgObjectBase bind(Object argumentSet, Object shreddedArgumentSet, Object valueSet)
                    throws UnsupportedMessageException, ArityException, UnsupportedTypeException {
        return new MapBoundArgObjectElement(index, parent.bind(argumentSet, shreddedArgumentSet, valueSet));
    }

    @Override
    public String toString() {
        return parent + "[" + index + "]";
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapBoundArgObjectElement extends MapBoundArgObjectBase {
    final MapBoundArgObjectBase parent;
    final long index;

    MapBoundArgObjectElement(long index, MapBoundArgObjectBase parent) {
        this.parent = parent;
        this.index = index;
    }

    @Override
    public String toString() {
        return parent + "[" + index + "]";
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary("this.parent") InteropLibrary parentInterop,
                    @CachedLibrary(limit = "2") InteropLibrary elementInterop) throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        if (arguments.length != 3) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(3, arguments.length);
        }
        Object value = parentInterop.execute(parent, arguments);
        try {
            return elementInterop.readArrayElement(value, index);
        } catch (UnsupportedMessageException | InvalidArrayIndexException e) {
            CompilerDirectives.transferToInterpreter();
            throw new MapException("cannot read element '" + index + "' from argument " + parent);
        }
    }
}

final class MapArgObjectMap extends MapArgObjectBase {
    final MapArgObjectBase parent;
    final Object function;

    MapArgObjectMap(Object function, MapArgObjectBase parent) {
        this.parent = parent;
        this.function = function;
    }

    @Override
    protected MapBoundArgObjectBase bind(Object argumentSet, Object shreddedArgumentSet, Object valueSet)
                    throws UnsupportedMessageException, ArityException, UnsupportedTypeException {
        return new MapBoundArgObjectMap(function, parent.bind(argumentSet, shreddedArgumentSet, valueSet));
    }

    @Override
    public String toString() {
        return parent + ".map(" + function.getClass().getSimpleName() + ")";
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapBoundArgObjectMap extends MapBoundArgObjectBase {
    final MapBoundArgObjectBase parent;
    final Object function;

    MapBoundArgObjectMap(Object function, MapBoundArgObjectBase parent) {
        this.parent = parent;
        this.function = function;
    }

    @Override
    public String toString() {
        return parent + ".map(" + function.getClass().getSimpleName() + ")";
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary("this.parent") InteropLibrary parentInterop,
                    @CachedLibrary("this.function") InteropLibrary mapInterop) throws UnsupportedTypeException, ArityException, UnsupportedMessageException {
        if (arguments.length != 3) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(3, arguments.length);
        }
        Object value = parentInterop.execute(parent, arguments);
        try {
            return mapInterop.execute(function, value);
        } catch (UnsupportedMessageException | UnsupportedTypeException | ArityException e) {
            CompilerDirectives.transferToInterpreter();
            throw new MapException("cannot map argument " + parent);
        }
    }
}

final class MapArgObjectShred extends MapArgObjectBase {
    final MapArgObjectBase parent;

    MapArgObjectShred(MapArgObjectBase parent) {
        this.parent = parent;
    }

    @Override
    protected MapBoundArgObjectBase bind(Object argumentSet, Object shreddedArgumentSet, Object valueSet)
                    throws UnsupportedMessageException, ArityException, UnsupportedTypeException {
        if (parent instanceof MapArgObjectArgument) {
            MapArgObjectArgument argument = (MapArgObjectArgument) parent;

            try {
                // read to make sure an index gets allocated
                INTEROP.asInt(INTEROP.readMember(argumentSet, argument.name));
            } catch (UnsupportedMessageException | UnknownIdentifierException e) {
                CompilerDirectives.transferToInterpreter();
                throw new MapException("cannot resolve argument index for '" + argument.name + "'");
            }
            int shreddedIndex;
            try {
                shreddedIndex = INTEROP.asInt(INTEROP.readMember(shreddedArgumentSet, argument.name));
            } catch (UnsupportedMessageException | UnknownIdentifierException e) {
                CompilerDirectives.transferToInterpreter();
                throw new MapException("cannot resolve shredded argument index for '" + argument.name + "'");
            }
            return new MapBoundShreddedArgObjectArgument(argument.name, shreddedIndex);

        } else {
            return new MapBoundArgObjectShred(parent.bind(argumentSet, shreddedArgumentSet, valueSet));
        }
    }

    @Override
    public String toString() {
        return parent + ".shred()";
    }
}

@ExportLibrary(InteropLibrary.class)
final class MapBoundArgObjectShred extends MapBoundArgObjectBase {
    final MapBoundArgObjectBase parent;

    MapBoundArgObjectShred(MapBoundArgObjectBase parent) {
        this.parent = parent;
    }

    @Override
    public String toString() {
        return parent + ".shred()";
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary("this.parent") InteropLibrary parentInterop) throws UnsupportedTypeException, ArityException, UnsupportedMessageException {
        if (arguments.length != 3) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(3, arguments.length);
        }
        return new ShreddedObject(parentInterop.execute(parent, arguments));
    }
}

@ExportLibrary(InteropLibrary.class)
public final class MappedFunction implements TruffleObject {

    final Object function;
    @CompilationFinal(dimensions = 1) final Object[] values;
    @CompilationFinal(dimensions = 1) final int[] shreddedArguments;
    final String description;
    final int valueCount;
    final Object returnValue;
    final Integer returnValueIndex;

    public MappedFunction(Object function, Object[] values, int[] shreddedArguments, int valueCount, Object returnValue, Integer returnValueIndex, String description) {
        this.function = function;
        this.values = values;
        this.shreddedArguments = shreddedArguments;
        this.valueCount = valueCount;
        this.returnValue = returnValue;
        this.returnValueIndex = returnValueIndex;
        this.description = description;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    @ExportMessage
    static final class Execute {

        static InteropLibrary[] createInterop(MappedFunction receiver) {
            InteropLibrary[] result = new InteropLibrary[receiver.values.length];
            for (int i = 0; i < result.length; i++) {
                result[i] = InteropLibrary.getFactory().create(receiver.values[i]);
            }
            return result;
        }

        @Specialization(guards = "cachedReceiver == receiver", limit = "1")
        static Object executeCached(@SuppressWarnings("unused") MappedFunction receiver, Object[] arguments,
                        @Cached("receiver") MappedFunction cachedReceiver,
                        @CachedLibrary("cachedReceiver.function") InteropLibrary functionInterop,
                        @CachedLibrary("cachedReceiver.returnValue") InteropLibrary resultInterop,
                        @Cached(value = "createInterop(cachedReceiver)") InteropLibrary[] valueInterop)
                        throws IllegalStateException, UnsupportedTypeException, ArityException, UnsupportedMessageException {
            Object[] shredded = createShredded(cachedReceiver.shreddedArguments, arguments);
            Object[] values = new Object[receiver.valueCount];
            ArgumentArray wrappedArguments = new ArgumentArray(arguments);
            ArgumentArray wrappedShreddedArguments = new ArgumentArray(shredded);
            ArgumentArray wrappedValueArguments = new ArgumentArray(values);
            Object[] mappedArguments = mapArguments(valueInterop, cachedReceiver, wrappedArguments, wrappedShreddedArguments, wrappedValueArguments);
            Object result = functionInterop.execute(cachedReceiver.function, mappedArguments);
            return processResult(receiver, resultInterop, values, wrappedArguments, wrappedShreddedArguments, wrappedValueArguments, result);
        }

        @ExplodeLoop
        private static Object[] mapArguments(InteropLibrary[] valueInterop, MappedFunction receiver, ArgumentArray arguments, ArgumentArray shredded, ArgumentArray values) {
            Object[] result = new Object[valueInterop.length];
            for (int i = 0; i < valueInterop.length; i++) {
                try {
                    result[i] = valueInterop[i].execute(receiver.values[i], arguments, shredded, values);
                } catch (UnsupportedTypeException | ArityException | UnsupportedMessageException e) {
                    CompilerDirectives.transferToInterpreter();
                    throw new MapException("cannot map parameter " + i + ": " + e.getMessage());
                }
            }
            return result;
        }

        @ExplodeLoop
        private static Object[] createShredded(int[] shreddedArguments, Object[] arguments) {
            Object[] result = new Object[shreddedArguments.length];
            for (int i = 0; i < shreddedArguments.length; i++) {
                result[i] = new ShreddedObject(arguments[shreddedArguments[i]]);
            }
            return result;
        }

        @Specialization(replaces = "executeCached")
        static Object execute(@SuppressWarnings("unused") MappedFunction receiver, Object[] arguments,
                        @CachedLibrary(limit = "2") InteropLibrary functionInterop,
                        @CachedLibrary(limit = "2") InteropLibrary valueInterop,
                        @CachedLibrary(limit = "2") InteropLibrary resultInterop)
                        throws IllegalStateException, UnsupportedTypeException, ArityException, UnsupportedMessageException {
            int[] shreddedArguments = receiver.shreddedArguments;
            Object[] shredded = new Object[shreddedArguments.length];
            for (int i = 0; i < shreddedArguments.length; i++) {
                shredded[i] = new ShreddedObject(arguments[shreddedArguments[i]]);
            }
            Object[] mappedArguments = new Object[receiver.values.length];
            Object[] values = new Object[receiver.valueCount];
            ArgumentArray wrappedArguments = new ArgumentArray(arguments);
            ArgumentArray wrappedShreddedArguments = new ArgumentArray(shredded);
            ArgumentArray wrappedValueArguments = new ArgumentArray(values);
            for (int i = 0; i < receiver.values.length; i++) {
                try {
                    mappedArguments[i] = valueInterop.execute(receiver.values[i], wrappedArguments, wrappedShreddedArguments, wrappedValueArguments);
                } catch (UnsupportedTypeException | ArityException | UnsupportedMessageException e) {
                    CompilerDirectives.transferToInterpreter();
                    throw new MapException("cannot map parameter " + i + ": " + e.getMessage());
                }
            }
            Object result = functionInterop.execute(receiver.function, mappedArguments);
            return processResult(receiver, resultInterop, values, wrappedArguments, wrappedShreddedArguments, wrappedValueArguments, result);
        }

        private static Object processResult(MappedFunction receiver, InteropLibrary resultInterop, Object[] values, ArgumentArray wrappedArguments, ArgumentArray wrappedShreddedArguments,
                        ArgumentArray wrappedValueArguments, Object result) throws UnsupportedTypeException, ArityException, UnsupportedMessageException {
            if (receiver.returnValueIndex != null) {
                values[receiver.returnValueIndex] = result;
            }
            return resultInterop.execute(receiver.returnValue, wrappedArguments, wrappedShreddedArguments, wrappedValueArguments);
        }
    }
}
