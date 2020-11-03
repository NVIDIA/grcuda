package com.nvidia.grcuda;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.Arrays;

@ExportLibrary(InteropLibrary.class)
public final class MemberSet implements TruffleObject {

    @CompilerDirectives.CompilationFinal(dimensions = 1) private final String[] values;

    public MemberSet(String... values) {
        this.values = values;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    public boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    public long getArraySize() {
        return values.length;
    }

    @ExportMessage
    public boolean isArrayElementReadable(long index) {
        return index >= 0 && index < values.length;
    }

    @ExportMessage
    public Object readArrayElement(long index) throws InvalidArrayIndexException {
        if ((index < 0) || (index >= values.length)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        return values[(int) index];
    }

    @CompilerDirectives.TruffleBoundary
    public boolean constainsValue(String name) {
        return Arrays.asList(values).contains(name);
    }
}