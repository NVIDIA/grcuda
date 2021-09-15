/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
package com.nvidia.grcuda;

import com.nvidia.grcuda.runtime.computation.ComputationArgument;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Collectors;

public abstract class Binding {
    protected final boolean hasCxxMangledName;
    protected final String name;
    protected final ComputationArgument[] computationArguments;
    protected String[] namespaceList;
    protected String mangledName;
    protected String libraryFileName;

    /**
     * Create a new binding.
     *
     * @param name a C style name or as fully qualified C++ name (e.g.,
     *            `namespace1::namespace2::name`).
     * @param computationArgumentList list of parameter names, types, and directions
     * @param hasCxxMangledName true if `name` is a C++ name and the symbol name is therefore
     *            mangled.
     */
    public Binding(String name, ArrayList<ComputationArgument> computationArgumentList, boolean hasCxxMangledName) {
        String[] identifierList = name.trim().split("::");
        this.name = identifierList[identifierList.length - 1];
        this.namespaceList = new String[identifierList.length - 1];
        if (identifierList.length > 1) {
            System.arraycopy(identifierList, 0, namespaceList, 0, identifierList.length - 1);
        }
        ComputationArgument[] params = new ComputationArgument[computationArgumentList.size()];
        this.computationArguments = computationArgumentList.toArray(params);
        this.hasCxxMangledName = hasCxxMangledName;
    }

    /**
     * Computes the mangled symbol using according to the g++/clang++ mangling rules including
     * substitutions. (see https://github.com/gchatelet/gcc_cpp_mangling_documentation)
     *
     * @return mangled symbol name.
     */
    public String getSymbolName() {
        if (!hasCxxMangledName) {
            // C symbol: no name mangling applied
            return name;
        }
        if (mangledName != null) {
            // return memoized name
            return mangledName;
        }

        // Mangle name with namespace and parameters
        String mangled = "_Z";
        // namespaces
        if (namespaceList.length > 0) {
            mangled += 'N';
            for (String namespaceName : namespaceList) {
                mangled += namespaceName.length() + namespaceName;
            }
            mangled += name.length() + name + 'E';
        } else {
            // symbol name
            mangled += name.length() + name;
        }
        // add arguments
        if (computationArguments.length == 0) {
            mangled += 'v';     // f() -> f(void) -> void
        } else {
            ArrayList<ComputationArgument> processedSymbolComputationArguments = new ArrayList<>(computationArguments.length);
            ArrayList<Integer> referencePositions = new ArrayList<>(computationArguments.length);
            int lastReference = 0;
            for (ComputationArgument currentParam : computationArguments) {
                if (currentParam.getKind() == ComputationArgument.Kind.BY_VALUE) {
                    // parameter of primitive type passed by-value: is not a symbol parameter
                    mangled += currentParam.getMangledType();
                } else {
                    // pointer parameter -> is a symbol parameter and subject to substitution rule
                    // -> check whether we've emitted a pointer of this (kind, type) already seen

                    boolean paramProcessed = false;
                    for (int i = 0; i < processedSymbolComputationArguments.size(); i++) {
                        ComputationArgument p = processedSymbolComputationArguments.get(i);
                        if (p.getKind() == currentParam.getKind() && p.getType() == currentParam.getType()) {
                            // found repetition -> apply substitution rule
                            int occurrencePos = referencePositions.get(i);
                            // encoding of substitution 0->S_, 1->S0_, 2->S1_, 3->S2_, etc.
                            mangled += (occurrencePos == 0) ? "S_" : ("S" + (occurrencePos - 1) + "_");
                            paramProcessed = true;
                            break;
                        }
                    }
                    if (!paramProcessed) {
                        // no repetition found -> no compression
                        mangled += currentParam.getMangledType();

                        // count "T*" as 1 symbol and "const T*" as 2 symbols
                        lastReference += currentParam.getKind() == ComputationArgument.Kind.POINTER_IN ? 2 : 1;
                        processedSymbolComputationArguments.add(currentParam);
                        referencePositions.add(lastReference - 1);
                    }
                }
            }
        }
        return mangled;
    }

    public void setNamespace(ArrayList<String> namespaceList) {
        this.namespaceList = new String[namespaceList.size()];
        this.namespaceList = namespaceList.toArray(this.namespaceList);
    }

    public void setLibraryFileName(String libraryFileName) {
        this.libraryFileName = libraryFileName;
    }

    public String getLibraryFileName() {
        return libraryFileName;
    }

    public String getNIDLParameterSignature() {
        return Arrays.stream(computationArguments).map(ComputationArgument::toNFISignatureElement).collect(Collectors.joining(", "));
    }

    public String toNIDLString() {
        return name + "(" + getNIDLParameterSignature() + ")";
    }

    @Override
    public String toString() {
        String argString = Arrays.stream(computationArguments).map(Object::toString).collect(Collectors.joining(", ", "[", "]"));
        return "Binding(name=" + name + ", argumentList=" + argString +
                        ", cxxnamespace=" + String.join("::", namespaceList) +
                        ", hasCxxMangledName=" + hasCxxMangledName + ", symbol=" + getSymbolName() + ")";
    }

    public String getName() {
        return name;
    }
}
