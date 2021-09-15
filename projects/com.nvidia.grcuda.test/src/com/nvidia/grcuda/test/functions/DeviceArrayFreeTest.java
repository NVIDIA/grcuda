/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
package com.nvidia.grcuda.test.functions;

import static org.junit.Assert.assertTrue;

import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotException;
import org.graalvm.polyglot.Value;
import org.junit.Test;

public class DeviceArrayFreeTest {

    //
    // Tests for 1-dim DeviceArray
    //

    @Test
    public void testCanInvokeFreeDeviceArray() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            // create DeviceArray
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            Value deviceArray = createDeviceArray.execute("int", 1000);
            assertTrue(deviceArray.canInvokeMember("free"));
            deviceArray.invokeMember("free");
            // check that freed flag set
            assertTrue(deviceArray.hasMember("isMemoryFreed"));
            assertTrue(deviceArray.getMember("isMemoryFreed").asBoolean());
        }
    }

    @Test(expected = PolyglotException.class)
    public void testDeviceArrayAccessAfterFreeThrows() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            // create DeviceArray
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            Value deviceArray = createDeviceArray.execute("int", 1000);
            deviceArray.invokeMember("free");
            deviceArray.setArrayElement(0, 42); // throws
        }
    }

    @Test(expected = PolyglotException.class)
    public void testDeviceArrayDoubleFreeThrows() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            // create DeviceArray
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            Value deviceArray = createDeviceArray.execute("int", 1000);
            deviceArray.invokeMember("free");
            deviceArray.invokeMember("free"); // throws
        }
    }

    //
    // Tests for Multi-dimensional DeviceArray
    //

    @Test
    public void testCanInvokeFreeMultiDimDeviceArray() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            // create DeviceArray
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            Value deviceArray = createDeviceArray.execute("int", 100, 100);
            assertTrue(deviceArray.canInvokeMember("free"));
            deviceArray.invokeMember("free");
            // check that freed flag set
            assertTrue(deviceArray.hasMember("isMemoryFreed"));
            assertTrue(deviceArray.getMember("isMemoryFreed").asBoolean());
        }
    }

    @Test(expected = PolyglotException.class)
    public void testMultiDimDeviceArrayAccessAfterFreeThrows() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            // create DeviceArray
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            Value deviceArray = createDeviceArray.execute("int", 100, 100);
            deviceArray.invokeMember("free");
            deviceArray.getArrayElement(0).setArrayElement(0, 42); // throws
        }
    }

    @Test(expected = PolyglotException.class)
    public void testMultiDimDeviceArrayDoubleFreeThrows() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            // create DeviceArray
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            Value deviceArray = createDeviceArray.execute("int", 100, 100);
            deviceArray.invokeMember("free");
            deviceArray.invokeMember("free"); // throws
        }
    }

}
