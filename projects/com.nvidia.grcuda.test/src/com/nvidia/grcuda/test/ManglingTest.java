/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import static org.junit.Assert.assertEquals;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import com.nvidia.grcuda.Binding;
import com.nvidia.grcuda.parser.antlr.NIDLParser;

public class ManglingTest {

    private static final int NUM_FUNCTIONS = 100;
    private static final int NUM_FUNCTION_PARAMS = 10;

    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    @Test
    public void testMangledNames() throws Exception {
        File nidlFile = tempFolder.newFile("host_functions.nidl");
        // Generate empty C++ functions with corresponding NIDL file
        CxxFunctionGenerator sourceGen = new CxxFunctionGenerator();
        String cxxSource = sourceGen.generateCxx(NUM_FUNCTIONS, NUM_FUNCTION_PARAMS, nidlFile);

        // Parse binding from NIDL file
        ArrayList<Binding> bindings = NIDLParser.parseNIDLFile(nidlFile.getAbsolutePath());

        // Copy c++ function using the host g++ and extract the mangled symbol names.
        Process compiler = Runtime.getRuntime().exec("g++ -x c++ -S -o - -");
        OutputStreamWriter stdin = new OutputStreamWriter(compiler.getOutputStream());
        BufferedReader stdout = new BufferedReader(new InputStreamReader(compiler.getInputStream()));
        BufferedReader stderr = new BufferedReader(new InputStreamReader(compiler.getErrorStream()));
        stdin.append(cxxSource);
        stdin.close();
        int cxxReturnCode = compiler.waitFor();
        if (cxxReturnCode != 0) {
            stderr.lines().forEach(System.out::println);
            throw new Exception("g++ return code != 0");
        }
        sourceGen.extractMangledNames(stdout);

        // Check GrCUDA-generated mangled names with compiler output
        int idx = 0;
        for (Binding binding : bindings) {
            String expectedMangledName = sourceGen.getMangledName(idx);
            String actualMangledName = binding.getSymbolName();
            assertEquals("incorrect mangling " + binding.toNIDLString(), expectedMangledName, actualMangledName);
            idx += 1;
        }
    }

    private static class CxxFunctionGenerator {
        private Random rand = new Random(0xdeadbeef);
        private String[] nidlSignatures;
        private String[] mangledNames;

        private static final String[] CXX_TYPES = {"bool", "char", "signed char", "unsigned char",
                        "short", "unsigned short", "int", "unsigned int", "long", "unsigned long",
                        "long long", "unsigned long long", "float", "double"};
        private static final String[] NIDL_TYPES = {"bool", "char", "sint8", "uint8",
                        "sint16", "uint16", "sint32", "uint32", "sint64", "uint64", "sll64",
                        "ull64", "float", "double"};
        private static final String[] NIDL_KIND = {"", "out pointer ", "in pointer "};

        String generateCxx(int numFunctions, int numFunctionParams, File nidlFile) throws IOException {
            nidlSignatures = new String[numFunctions];
            mangledNames = new String[numFunctions];
            StringBuffer cxxSourceBuffer = new StringBuffer();

            class Arg {
                int seqId;
                int kindId;
                int typeId;

                Arg(int seq) {
                    seqId = seq;
                    kindId = rand.nextInt(3);
                    typeId = rand.nextInt(CXX_TYPES.length);
                }
            }

            for (int functionId = 0; functionId < numFunctions; functionId++) {
                String functionName = String.format("function%03d", functionId);
                Arg[] args;
                if (functionId == 0) {
                    // let first function take no parameters
                    args = new Arg[0];
                } else {
                    args = IntStream.range(0, numFunctionParams).mapToObj(i -> new Arg(i)).toArray(count -> new Arg[count]);
                }

                // prepare C++ function
                cxxSourceBuffer.append("void ");
                cxxSourceBuffer.append(functionName);
                cxxSourceBuffer.append("(");
                for (Arg arg : args) {
                    switch (arg.kindId) {
                        case 0:
                            cxxSourceBuffer.append(CXX_TYPES[arg.typeId]);
                            break;
                        case 1:
                            cxxSourceBuffer.append(CXX_TYPES[arg.typeId]);
                            cxxSourceBuffer.append("*");
                            break;
                        default:
                            cxxSourceBuffer.append("const ");
                            cxxSourceBuffer.append(CXX_TYPES[arg.typeId]);
                            cxxSourceBuffer.append("*");
                            break;
                    }
                    cxxSourceBuffer.append(" ");
                    cxxSourceBuffer.append("arg" + arg.seqId);
                    if (arg.seqId < (args.length - 1)) {
                        cxxSourceBuffer.append(", ");
                    }
                }
                cxxSourceBuffer.append(") { }\n");

                // prepare NIDL
                StringBuffer nidlBuffer = new StringBuffer();
                nidlBuffer.append(functionName);
                nidlBuffer.append("(");
                for (Arg arg : args) {
                    nidlBuffer.append("arg" + arg.seqId);
                    nidlBuffer.append(": ");
                    nidlBuffer.append(NIDL_KIND[arg.kindId]);
                    nidlBuffer.append(NIDL_TYPES[arg.typeId]);
                    if (arg.seqId < (args.length - 1)) {
                        nidlBuffer.append(", ");
                    }
                }
                nidlBuffer.append("): void");
                nidlSignatures[functionId] = nidlBuffer.toString();
            }

            PrintWriter w = new PrintWriter(new FileWriter(nidlFile));
            w.println("hostfuncs {");
            for (String sig : nidlSignatures) {
                w.println(sig);
            }
            w.println("}");
            w.close();

            return cxxSourceBuffer.toString();
        }

        public void extractMangledNames(BufferedReader stdout) {
            Predicate<String> filterSymbolLines = Pattern.compile("^_Z").asPredicate();
            mangledNames = stdout.lines().filter(filterSymbolLines).map(s -> s.substring(0, s.indexOf(':'))).toArray(count -> new String[count]);
            assert mangledNames.length == nidlSignatures.length : "number of mangled names differs from expectation";
        }

        String getMangledName(int idx) {
            return mangledNames[idx];
        }
    }
}
