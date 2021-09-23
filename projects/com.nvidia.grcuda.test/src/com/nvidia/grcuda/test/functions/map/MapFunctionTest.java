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
package com.nvidia.grcuda.test.functions.map;

import java.lang.reflect.Array;

import org.junit.Assert;
import org.junit.Test;

import com.nvidia.grcuda.functions.map.MapArgObject;
import com.nvidia.grcuda.functions.map.MapFunction;
import com.nvidia.grcuda.functions.map.MappedFunction;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.test.host.ProxyLanguageEnvTest;

public class MapFunctionTest extends ProxyLanguageEnvTest {

    private static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    interface CheckFunction {
        boolean check(Object value);
    }

    static final class TestContainer implements TruffleObject {
        final Object[] values;

        TestContainer(Object[] values) {
            this.values = values;
        }

        public static void check(Object value, CheckFunction... tests) {
            Assert.assertTrue(value instanceof TestContainer);
            Object[] values = ((TestContainer) value).values;
            Assert.assertEquals(values.length, tests.length);
            for (int i = 0; i < values.length; i++) {
                Assert.assertTrue(tests[i].check(values[i]));
            }
        }
    }

    @ExportLibrary(InteropLibrary.class)
    static final class TestFunction implements TruffleObject {
        @ExportMessage
        @SuppressWarnings("static-method")
        boolean isExecutable() {
            return true;
        }

        @ExportMessage
        @SuppressWarnings("static-method")
        Object execute(Object[] arguments) {
            return new TestContainer(arguments);
        }
    }

    @ExportLibrary(InteropLibrary.class)
    final class TestNewArrayFunction implements TruffleObject {
        @ExportMessage
        @SuppressWarnings("static-method")
        boolean isExecutable() {
            return true;
        }

        @ExportMessage
        @SuppressWarnings("static-method")
        Object execute(Object[] arguments) {
            Assert.assertEquals(1, arguments.length);
            return asTruffleObject(new int[(int) (long) arguments[0]]);
        }
    }

    @ExportLibrary(InteropLibrary.class)
    static final class TestFunction2 implements TruffleObject {
        @ExportMessage
        @SuppressWarnings("static-method")
        boolean isExecutable() {
            return true;
        }

        @ExportMessage
        @SuppressWarnings("static-method")
        Object execute(Object[] arguments) {
            return arguments.length;
        }
    }

    public static final class TestClass {
        public int a;
        public double b;
        public Object c;
        public Object d;

        public TestClass(int a, double b, Object c, Object d) {
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;
        }
    }

    public static final class TestClass2 {
        public int c;
        public double d;

        public TestClass2(int c, double d) {
            this.c = c;
            this.d = d;
        }
    }

    private MapFunction map = new MapFunction();

    @Test
    public void testArgCount() throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        Object mapped = map.map(new TestFunction());
        TestContainer.check(INTEROP.execute(mapped));
        mapped = map.map(new TestFunction(), map.arg("a"));
        TestContainer.check(INTEROP.execute(mapped, 1.0), v -> (double) v == 1);
        mapped = map.map(new TestFunction(), map.arg("a"), map.arg("b"));
        TestContainer.check(INTEROP.execute(mapped, 1, 2), v -> (int) v == 1, v -> (int) v == 2);
    }

    @Test
    public void testArgReuse() throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        Object mapped = map.map(new TestFunction());
        TestContainer.check(INTEROP.execute(mapped));
        mapped = map.map(new TestFunction(), map.arg("a"), map.arg("a"));
        TestContainer.check(INTEROP.execute(mapped, 1.0), v -> (double) v == 1, v -> (double) v == 1);
        mapped = map.map(new TestFunction(), map.arg("a"), map.arg("b"));
        TestContainer.check(INTEROP.execute(mapped, 1, 2), v -> (int) v == 1, v -> (int) v == 2);
    }

    private static boolean checkArray(Object value, Object array) {
        try {
            Assert.assertEquals(Array.getLength(array), INTEROP.getArraySize(value));
            for (int i = 0; i < Array.getLength(array); i++) {
                Assert.assertEquals(Array.get(array, i), INTEROP.readArrayElement(value, i));
            }
        } catch (UnsupportedMessageException | IllegalArgumentException | ArrayIndexOutOfBoundsException | InvalidArrayIndexException e) {
            throw new AssertionError(e);
        }
        return true;
    }

    @Test
    public void testSimple() throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        Object intArray = asTruffleObject(new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        Object doubleArray = asTruffleObject(new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        Object object = asTruffleObject(new TestClass(5, 10, new int[]{0, 1, 2, 3, 4}, new double[]{0, 1, 2, 3, 4}));

        MappedFunction mapped = map.map(new TestFunction(), map.arg("a").readArrayElement(2));
        TestContainer.check(INTEROP.execute(mapped, intArray), v -> (int) v == 2);
        mapped = map.map(new TestFunction(), map.arg("a").readArrayElement(2), map.arg("a").readArrayElement(4));
        TestContainer.check(INTEROP.execute(mapped, doubleArray), v -> (double) v == 2, v -> (double) v == 4);

        mapped = map.map(new TestFunction(), map.arg("a").readMember("a"), map.arg("a").readMember("c").readArrayElement(3));
        TestContainer.check(INTEROP.execute(mapped, object), v -> (int) v == 5, v -> (int) v == 3);
    }

    @Test
    public void testShred() throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        Object object = asTruffleObject(new TestClass(5, 10, new int[]{0, 1, 2, 3, 4}, new double[]{0, 1, 2, 3, 4}));
        Object object2 = asTruffleObject(new Object[]{new TestClass2(5, 10), new TestClass2(6, 11), new TestClass2(7, 12), new TestClass2(8, 13)});

        MapArgObject argC = map.arg("a").shred().readMember("c");
        MapArgObject argD = map.arg("a").shred().readMember("d");
        MappedFunction mapped = map.map(new TestFunction(), argC, argD, map.size(argC, argD));
        TestContainer.check(INTEROP.execute(mapped, object), v -> checkArray(v, new int[]{0, 1, 2, 3, 4}), v -> checkArray(v, new double[]{0, 1, 2, 3, 4}), v -> (long) v == 5);

        TestContainer.check(INTEROP.execute(mapped, object2), v -> checkArray(v, new int[]{5, 6, 7, 8}), v -> checkArray(v, new double[]{10, 11, 12, 13}), v -> (long) v == 4);
    }

    @Test
    public void testValue() throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        Object mapped = map.map(new TestFunction(), map.value("foo", new TestFunction2(), map.value("bar", new TestFunction2())), map.value("foo"));
        TestContainer.check(INTEROP.execute(mapped), v -> (int) v == 1, v -> (int) v == 1);
    }

    @Test
    public void testReturn() throws ArityException, UnsupportedTypeException, UnsupportedMessageException {
        Object object = asTruffleObject(new TestClass(5, 10, new int[]{0, 1, 2, 3, 4}, new double[]{0, 1, 2, 3, 4}));
        Object object2 = asTruffleObject(new Object[]{new TestClass2(5, 10), new TestClass2(6, 11), new TestClass2(7, 12), new TestClass2(8, 13)});

        MapArgObject argC = map.arg("a").shred().readMember("c");
        MapArgObject argD = map.arg("a").shred().readMember("d");
        MappedFunction mapped = map.ret(map.value("output")).map(new TestFunction(), argC, argD, map.value("output", new TestNewArrayFunction(), map.size(argC, argD)));
        checkArray(INTEROP.execute(mapped, object), new int[]{0, 0, 0, 0, 0});
        checkArray(INTEROP.execute(mapped, object2), new int[]{0, 0, 0, 0});
    }
}
