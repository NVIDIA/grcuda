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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class BindKernelTest {

    /** CUDA C source code of incrementing kernel. */
    private static final String INCREMENT_KERNEL_SOURCE = "extern \"C\"                                  \n" +
                    "__global__ void inc_kernel(int *out_arr, const int *in_arr, size_t num_elements) {  \n" +
                    "  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;         \n" +
                    "       idx += gridDim.x * blockDim.x) {                                             \n" +
                    "    out_arr[idx] = in_arr[idx] + 1;                                                 \n" +
                    "  }                                                                                 \n" +
                    "}\n";

    /** NFI signature of incrementing kernel. */
    private static final String INCREMENT_KERNEL_SIGNATURE = "pointer, pointer, uint64";

    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    @Test
    public void testBindKernel() throws IOException, InterruptedException {
        // Write CUDA C source file
        File sourceFile = tempFolder.newFile("inc_kernel.cu");
        PrintWriter writer = new PrintWriter(new FileWriter(sourceFile));
        writer.write(INCREMENT_KERNEL_SOURCE);
        writer.close();
        String cubinFileName = sourceFile.getParent() + File.separator + "inc_kernel.ptx";

        // Compile source file with NVCC
        Process compiler = Runtime.getRuntime().exec("nvcc --ptx " +
                        sourceFile.getAbsolutePath() + " -o " + cubinFileName);
        BufferedReader output = new BufferedReader(new InputStreamReader(compiler.getErrorStream()));
        int nvccReturnCode = compiler.waitFor();
        output.lines().forEach(System.out::println);
        assertEquals(0, nvccReturnCode);

        // Build inc_kernel symbol, launch it, and check results.
        try (Context context = Context.newBuilder().allowAllAccess(true).build()) {
            final int numElements = 1000;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value bindkernel = context.eval("grcuda", "bindkernel");
            Value incrKernel = bindkernel.execute(cubinFileName, "inc_kernel",
                            INCREMENT_KERNEL_SIGNATURE);
            assertNotNull(incrKernel);
            assertTrue(incrKernel.canExecute());
            assertEquals(0, incrKernel.getMember("launchCount").asInt());
            assertNotNull(incrKernel.getMember("ptx").asString());
            Value inDevArray = deviceArrayConstructor.execute("int", numElements);
            Value outDevArray = deviceArrayConstructor.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                inDevArray.setArrayElement(i, i);
                outDevArray.setArrayElement(i, 0);
            }
            Value configuredIncKernel = incrKernel.execute(8, 128);  // <<<8, 128>>> 8 blocks a 128
                                                                     // threads
            assertTrue(configuredIncKernel.canExecute());
            configuredIncKernel.execute(outDevArray, inDevArray, numElements);
            // implicit synchronization

            // verify result
            for (int i = 0; i < numElements; ++i) {
                assertEquals(i, inDevArray.getArrayElement(i).asInt());
                assertEquals(i + 1, outDevArray.getArrayElement(i).asInt());
            }
            assertEquals(1, incrKernel.getMember("launchCount").asInt());
        }
    }
}
