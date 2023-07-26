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
package com.nvidia.grcuda.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.nvidia.grcuda.GrCUDAOptionMap;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.cudalibraries.cublas.CUBLASRegistry;
import com.nvidia.grcuda.cudalibraries.cuml.CUMLRegistry;
import com.nvidia.grcuda.cudalibraries.tensorrt.TensorRTRegistry;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import com.nvidia.grcuda.test.util.mock.OptionValuesMock;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.StopIterationException;
import com.oracle.truffle.api.interop.UnknownKeyException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionValues;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;

import java.util.HashMap;

public class GrCUDAOptionMapTest {

    private GrCUDAOptionMap optionMap;
    private OptionValues optionValues;
    public void initializeDefault() {
        optionValues = new OptionValuesMock();
        setOption(GrCUDAOptions.CuBLASEnabled, true);
        setOption(GrCUDAOptions.CuMLEnabled, true);
        setOption(GrCUDAOptions.ForceStreamAttach, GrCUDAOptionMap.DEFAULT_FORCE_STREAM_ATTACH);
        setOption(GrCUDAOptions.InputPrefetch, false);
        setOption(GrCUDAOptions.TensorRTEnabled, false);
        setOption(GrCUDAOptions.CuBLASLibrary, CUBLASRegistry.DEFAULT_LIBRARY);
        setOption(GrCUDAOptions.CuMLLibrary, CUMLRegistry.DEFAULT_LIBRARY);
        setOption(GrCUDAOptions.ExecutionPolicy, GrCUDAOptionMap.DEFAULT_EXECUTION_POLICY.toString());
        setOption(GrCUDAOptions.DependencyPolicy, GrCUDAOptionMap.DEFAULT_DEPENDENCY_POLICY.toString());
        setOption(GrCUDAOptions.RetrieveNewStreamPolicy, GrCUDAOptionMap.DEFAULT_RETRIEVE_STREAM_POLICY.toString());
        setOption(GrCUDAOptions.RetrieveParentStreamPolicy, GrCUDAOptionMap.DEFAULT_PARENT_STREAM_POLICY.toString());
        setOption(GrCUDAOptions.TensorRTLibrary, TensorRTRegistry.DEFAULT_LIBRARY);
        optionMap = new GrCUDAOptionMap(optionValues);
    }

    public void initializeNull() {
        optionValues = new OptionValuesMock();
        setOption(GrCUDAOptions.ExecutionPolicy, GrCUDAOptionMap.DEFAULT_EXECUTION_POLICY.toString());
        setOption(GrCUDAOptions.DependencyPolicy, GrCUDAOptionMap.DEFAULT_DEPENDENCY_POLICY.toString());
        setOption(GrCUDAOptions.RetrieveNewStreamPolicy, GrCUDAOptionMap.DEFAULT_RETRIEVE_STREAM_POLICY.toString());
        setOption(GrCUDAOptions.RetrieveParentStreamPolicy, GrCUDAOptionMap.DEFAULT_PARENT_STREAM_POLICY.toString());
        setOption(GrCUDAOptions.TensorRTLibrary, null);
        optionMap = new GrCUDAOptionMap(optionValues);
    }

    private <T> void setOption(OptionKey<T> key, T value) {
        optionValues.set(key, value);
    }

    @Test
    public void testGetOption(){
        initializeDefault();
        assertEquals(optionMap.isCuBLASEnabled(), true);
        assertEquals(optionMap.isForceStreamAttach(), false);
        assertEquals(optionMap.getCuBLASLibrary(), CUBLASRegistry.DEFAULT_LIBRARY);
        assertEquals(optionMap.getDependencyPolicy(), GrCUDAOptionMap.DEFAULT_DEPENDENCY_POLICY);
    }

    @Test(expected = UnknownKeyException.class)
    public void testReadUnknownKey() throws UnsupportedMessageException, UnknownKeyException {
        initializeDefault();
        optionMap.readHashValue("NotPresent");
    }

    @Test(expected = UnsupportedMessageException.class)
    public void testReadUnsupportedMessage() throws UnsupportedMessageException, UnknownKeyException {
        initializeDefault();
        optionMap.readHashValue(null);
    }

    @Test
    public void testGetHashEntriesIterator(){
        initializeDefault();
        GrCUDAOptionMap.EntriesIterator hashIterator = (GrCUDAOptionMap.EntriesIterator) optionMap.getHashEntriesIterator();
        optionMap.getOptions().forEach((key, value) -> {
            assertTrue(hashIterator.hasIteratorNextElement());
            try {
                GrCUDAOptionMap.GrCUDAOptionTuple elem = hashIterator.getIteratorNextElement();
                assertEquals(key, elem.readArrayElement(0));
                assertEquals(value.toString(), elem.readArrayElement(1));
            } catch (StopIterationException | InvalidArrayIndexException e) {
                e.printStackTrace();
            }
        });
    }

    @Test(expected = StopIterationException.class)
    public void testGetStopIteration() throws StopIterationException {
        initializeDefault();
        GrCUDAOptionMap.EntriesIterator hashIterator = (GrCUDAOptionMap.EntriesIterator) optionMap.getHashEntriesIterator();
        do {
            try {
                hashIterator.getIteratorNextElement();
            } catch (StopIterationException e) {
                e.printStackTrace();
            }
        } while (hashIterator.hasIteratorNextElement());
        hashIterator.getIteratorNextElement();
    }

    @Test(expected = NullPointerException.class)
    public void testGetNullPointerExceptionWhenRetrievingValue() throws NullPointerException, StopIterationException {
        initializeNull();
        GrCUDAOptionMap.EntriesIterator hashIterator = (GrCUDAOptionMap.EntriesIterator) optionMap.getHashEntriesIterator();
        do {
            hashIterator.getIteratorNextElement();
        } while (hashIterator.hasIteratorNextElement());
    }

    @Test(expected = InvalidArrayIndexException.class)
    public void testGetInvalidIndex() throws InvalidArrayIndexException {
        initializeDefault();
        GrCUDAOptionMap.EntriesIterator hashIterator = (GrCUDAOptionMap.EntriesIterator) optionMap.getHashEntriesIterator();
        try {
            GrCUDAOptionMap.GrCUDAOptionTuple elem = hashIterator.getIteratorNextElement();
            assertEquals(2, elem.getArraySize());
            assertFalse(elem.isArrayElementReadable(2));
            elem.readArrayElement(2);
        } catch (StopIterationException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testGetOptionsFunction() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", ExecutionPolicyEnum.ASYNC.toString()).option("grcuda.EnableComputationTimers", "true").build()) {
            // Obtain the options map;
            Value options = ctx.eval("grcuda", "getoptions").execute();
            // Check the we have a map;
            assertTrue(options.hasHashEntries());

            // Obtain some options;
            assertEquals(ExecutionPolicyEnum.ASYNC.toString(), options.getHashValue("grcuda.ExecutionPolicy").asString());
            assertTrue(Boolean.parseBoolean(options.getHashValue("grcuda.EnableComputationTimers").asString()));
            assertFalse(Boolean.parseBoolean(options.getHashValue("grcuda.ForceStreamAttach").asString()));
            assertEquals(GrCUDAOptionMap.DEFAULT_NUMBER_OF_GPUs, Integer.valueOf(options.getHashValue("grcuda.NumberOfGPUs").asString()));
        }
    }

    @Test
    public void testGetOptionsFunctionIterator() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", ExecutionPolicyEnum.ASYNC.toString()).build()) {
            // Obtain the options map;
            Value options = ctx.eval("grcuda", "getoptions").execute();
            // Get the iterator;
            Value iterator = options.getHashEntriesIterator();
            int optionCount = 0;
            // Check that we can find a specific option key and value;
            String optionKeyToFind = "grcuda.ExecutionPolicy";
            String optionValueToFind = ExecutionPolicyEnum.ASYNC.toString();
            boolean optionFound = false;
            while (iterator.hasIteratorNextElement()) {
                Value option = iterator.getIteratorNextElement();
                assertEquals(2, option.getArraySize());
                if (option.getArrayElement(0).asString().equals(optionKeyToFind)) {
                    optionFound = true;
                    assertEquals(optionValueToFind, option.getArrayElement(1).asString());
                }
                optionCount++;
            }
            assertTrue(optionFound);
            assertTrue(iterator.isIterator());
            assertFalse(iterator.hasIteratorNextElement());
            assertEquals(options.getHashSize(), optionCount);
        }
    }
}
