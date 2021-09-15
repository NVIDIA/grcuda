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
package com.nvidia.grcuda.test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertEquals;

import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class BindTest {
    /** CUDA C source code of incrementing kernel. */
    private static final String CXX_SOURCE = "                                                       \n" +
                    "// C kernels \n" +
                    "__global__ void inc_kernel(float *out_arr, const int *in_arr, int num_elements) { \n" +
                    "  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; \n" +
                    "       idx += gridDim.x * blockDim.x) { \n" +
                    "    out_arr[idx] = in_arr[idx] + 1.0f; \n" +
                    "  } \n" +
                    "} \n" +
                    "__global__ void inc_inplace_kernel(int *inout_arr, int num_elements) { \n" +
                    "  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; \n" +
                    "       idx += gridDim.x * blockDim.x) { \n" +
                    "    inout_arr[idx] += 1; \n" +
                    "  } \n" +
                    "} \n" +
                    "\n" +
                    "// C functions \n" +
                    "extern \"C\" int inc_host(int blocks, int threads_per_block, \n" +
                    "                          float *out_arr, const int *in_arr, int num_elements) { \n" +
                    "  inc_kernel<<<blocks,    threads_per_block>>>(out_arr, in_arr, num_elements); \n" +
                    "  return cudaDeviceSynchronize(); \n" +
                    "} \n" +
                    "extern \"C\" int inc_inplace_host(int blocks, int threads_per_block, \n" +
                    "                                  int *inout_arr, int num_elements) { \n" +
                    "  inc_inplace_kernel<<<blocks, threads_per_block>>>(inout_arr, num_elements); \n" +
                    "  return cudaDeviceSynchronize(); \n" +
                    "} \n" +
                    "// C++ functions \n" +
                    "int cxx_inc_host(int blocks, int threads_per_block, \n" +
                    "                 float *out_arr, const int *in_arr, int num_elements) { \n" +
                    "  inc_kernel<<<blocks, threads_per_block>>>(out_arr, in_arr, num_elements); \n" +
                    "  return cudaDeviceSynchronize(); \n" +
                    "} \n" +
                    "int cxx_inc_inplace_host(int blocks, int threads_per_block, \n" +
                    "                         int *inout_arr, int num_elements) { \n" +
                    "  inc_inplace_kernel<<<blocks, threads_per_block>>>(inout_arr, num_elements); \n" +
                    "  return cudaDeviceSynchronize(); \n" +
                    "}\n";
    private static final int numElements = 100;

    @ClassRule public static TemporaryFolder tempFolder = new TemporaryFolder();
    public static String dynamicLibraryFile;

    @BeforeClass
    public static void setupUpClass() throws IOException, InterruptedException {
        // Write CUDA C source file
        File sourceFile = tempFolder.newFile("inc_kernel.cu");
        PrintWriter writer = new PrintWriter(new FileWriter(sourceFile));
        writer.write(CXX_SOURCE);
        writer.close();
        dynamicLibraryFile = sourceFile.getParent() + File.separator + "libfoo.so";

        // Compile source file with NVCC
        Process compiler = Runtime.getRuntime().exec("nvcc -shared -Xcompiler -fPIC " +
                        sourceFile.getAbsolutePath() + " -o " + dynamicLibraryFile);
        BufferedReader output = new BufferedReader(new InputStreamReader(compiler.getErrorStream()));
        int nvccReturnCode = compiler.waitFor();
        output.lines().forEach(System.out::println);
        assertEquals(0, nvccReturnCode);
    }

    public void callWithInAndOutArguments(String... bindArgs) {
        try (Context polyglot = GrCUDATestUtil.buildTestContext().build()) {
            Value cu = polyglot.eval("grcuda", "CU");
            Value inDeviceArray = cu.getMember("DeviceArray").execute("int", numElements);
            Value outDeviceArray = cu.getMember("DeviceArray").execute("float", numElements);
            for (int i = 0; i < numElements; i++) {
                inDeviceArray.setArrayElement(i, Integer.valueOf(i));
                outDeviceArray.setArrayElement(i, Float.valueOf(0));
            }

            // get function from shared library
            Value bind = cu.getMember("bind");
            Value function = bindArgs.length > 1 ? bind.execute(dynamicLibraryFile, bindArgs[0], bindArgs[1])
                            : bind.execute(dynamicLibraryFile, bindArgs[0]);
            assertNotNull(function);

            // call function
            int blocks = 80;
            int threadsPerBlock = 256;
            function.execute(blocks, threadsPerBlock, outDeviceArray, inDeviceArray, numElements);

            // verify result
            for (int i = 0; i < numElements; i++) {
                assertEquals(i + 1.0f, outDeviceArray.getArrayElement(i).asFloat(), 1e-3f);
            }
        }
    }

    public void callWithInoutArgument(String... bindArgs) {
        try (Context polyglot = GrCUDATestUtil.buildTestContext().build()) {
            Value cu = polyglot.eval("grcuda", "CU");
            Value inoutDeviceArray = cu.getMember("DeviceArray").execute("int", numElements);
            for (int i = 0; i < numElements; i++) {
                inoutDeviceArray.setArrayElement(i, Integer.valueOf(i));
            }

            // get function from shared library
            Value bind = cu.getMember("bind");
            Value function = bindArgs.length > 1 ? bind.execute(dynamicLibraryFile, bindArgs[0], bindArgs[1])
                            : bind.execute(dynamicLibraryFile, bindArgs[0]);
            assertNotNull(function);

            // call function
            int blocks = 80;
            int threadsPerBlock = 256;
            function.execute(blocks, threadsPerBlock, inoutDeviceArray, numElements);

            // verify result
            for (int i = 0; i < numElements; i++) {
                assertEquals(i + 1, inoutDeviceArray.getArrayElement(i).asInt());
            }
        }
    }

    @Test
    public void testCcallLegacyNFISignatureWithInAndOutArguments() {
        callWithInAndOutArguments("inc_host", "(sint32, sint32, pointer, pointer, sint32): sint32");
    }

    @Test
    public void testCcallLegacyNFISignatureWithInoutArgument() {
        callWithInoutArgument("inc_inplace_host", "(sint32, sint32, pointer, sint32): sint32");
    }

    @Test
    public void testCxxCallLegacyNFISignatureWithInAndOutArguments() {
        callWithInAndOutArguments("_Z12cxx_inc_hostiiPfPKii", "(sint32, sint32, pointer, pointer, sint32): sint32");
    }

    @Test
    public void testCxxCallLegacyNFISignatureWithInoutArgument() {
        callWithInoutArgument("_Z20cxx_inc_inplace_hostiiPii", "(sint32, sint32, pointer, sint32): sint32");
    }

    @Test
    public void testCcallNIDLSignatureWithInAndOutArguments() {
        callWithInAndOutArguments("" +
                        "inc_host(blocks: sint32, threads_per_block: sint32, out_arr: out pointer float, " +
                        "in_arr: in pointer sint32, num_elements: sint32): sint32");
    }

    @Test
    public void testCcallNIDLSignatureWithInoutArguments() {
        callWithInoutArgument("" +
                        "inc_inplace_host(blocks: sint32, threads_per_block: sint32, inout_arr: inout pointer sint32, " +
                        "num_elements: sint32): sint32");
    }

    @Test
    public void testCxxCallNIDLSignatureWithInAndOutArguments() {
        callWithInAndOutArguments("cxx " +
                        "cxx_inc_host(blocks: sint32, threads_per_block: sint32, out_arr: out pointer float, " +
                        "in_arr: in pointer sint32, num_elements: sint32): sint32");
    }

    @Test
    public void testCxxcallNIDLSignatureWithInoutArguments() {
        callWithInoutArgument("cxx " +
                        "cxx_inc_inplace_host(blocks: sint32, threads_per_block: sint32, inout_arr: inout pointer sint32, " +
                        "num_elements: sint32): sint32");
    }
}
