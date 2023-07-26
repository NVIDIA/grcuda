/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
package com.nvidia.grcuda.test.runtime.array;

import java.util.Arrays;
import java.util.Collection;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotAccess;
import org.graalvm.polyglot.Value;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class DeviceArrayTest {
    @Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
                        {"char", (byte) 42, 100},
                        {"short", (short) 42, 100},
                        {"int", 42, 100},
                        {"long", 42L, 100},
                        {"float", 42.0f, 100},
                        {"double", 42.0, 100},
        });
    }

    private final String dataTypeString;
    private final Object testValue;
    private final int arrayLength;

    public DeviceArrayTest(String dataTypeString, Object testValue, int arrayLength) {
        this.dataTypeString = dataTypeString;
        this.testValue = testValue;
        this.arrayLength = arrayLength;
    }

    @Test
    public void testDeviceArrayCreationFromArrayExpression() {
        try (Context context = GrCUDATestUtil.buildTestContext().allowPolyglotAccess(PolyglotAccess.ALL).build()) {
            Value deviceArray = context.eval("grcuda", dataTypeString + "[" + arrayLength + "]");
            assertTrue(deviceArray.hasArrayElements());
            assertEquals(arrayLength, deviceArray.getArraySize());
        }
    }

    @Test
    public void testDeviceArrayCreationFromDeviceArrayConstructor() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceArrayFunc = context.eval("grcuda", "DeviceArray");
            Value deviceArray = deviceArrayFunc.execute(dataTypeString, arrayLength);
            assertTrue(deviceArray.hasArrayElements());
            assertEquals(arrayLength, deviceArray.getArraySize());
        }
    }

    private void checkEquality(Object expected, Value actual) {
        switch (dataTypeString) {
            case "char":
                assertTrue(actual.fitsInByte());
                assertEquals(((Number) expected).byteValue(), actual.asByte());
                break;
            case "short":
                assertTrue(actual.fitsInShort());
                assertEquals(((Number) expected).shortValue(), actual.asShort());
                break;
            case "int":
                assertTrue(actual.fitsInInt());
                assertEquals(((Number) expected).intValue(), actual.asInt());
                break;
            case "long":
                assertTrue(actual.fitsInLong());
                assertEquals(((Number) expected).longValue(), actual.asLong());
                break;
            case "float":
                assertTrue(actual.fitsInFloat());
                assertEquals(((Number) expected).floatValue(), actual.asFloat(), 1e-6);
                break;
            case "double":
                assertTrue(actual.fitsInDouble());
                assertEquals(((Number) expected).doubleValue(), actual.asDouble(), 1e-6);
                break;
            default:
                throw new RuntimeException("invalid type " + dataTypeString);
        }
    }

    private void setElement(Value array, int index, Number value) {
        switch (dataTypeString) {
            case "char":
                array.setArrayElement(index, value.byteValue());
                break;
            case "short":
                array.setArrayElement(index, value.shortValue());
                break;
            case "int":
                array.setArrayElement(index, value.intValue());
                break;
            case "long":
                array.setArrayElement(index, value.longValue());
                break;
            case "float":
                array.setArrayElement(index, value.floatValue());
                break;
            case "double":
                array.setArrayElement(index, value.doubleValue());
                break;
            default:
                throw new RuntimeException("invalid type " + dataTypeString);
        }
    }

    @Test
    public void testDeviceArrayGetValue() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceArray = context.eval("grcuda", dataTypeString + "[" + arrayLength + "]");
            assertTrue(deviceArray.hasArrayElements());
            assertEquals(arrayLength, deviceArray.getArraySize());
            Value firstElement = deviceArray.getArrayElement(0);
            checkEquality(0, firstElement);
        }
    }

    @Test
    public void testDeviceArraySetAndGetsSetValue() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value deviceArray = context.eval("grcuda", dataTypeString + "[" + arrayLength + "]");
            assertTrue(deviceArray.hasArrayElements());
            assertEquals(arrayLength, deviceArray.getArraySize());
            final Number value = (Number) testValue;
            setElement(deviceArray, 0, value);
            Value firstElement = deviceArray.getArrayElement(0);
            checkEquality(value, firstElement);
        }
    }
}
