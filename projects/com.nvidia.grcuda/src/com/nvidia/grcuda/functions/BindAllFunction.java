/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.nvidia.grcuda.functions;

import java.util.ArrayList;
import java.util.Optional;
import java.util.regex.Pattern;

import com.nvidia.grcuda.Binding;
import com.nvidia.grcuda.FunctionBinding;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.KernelBinding;
import com.nvidia.grcuda.Namespace;
import com.nvidia.grcuda.runtime.LazyKernel;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.parser.antlr.NIDLParser;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class BindAllFunction extends Function {

    @SuppressWarnings("unused") private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;

    private final GrCUDAContext context;

    public BindAllFunction(GrCUDAContext context) {
        super("bindall");
        this.grCUDAExecutionContext = context.getGrCUDAExecutionContext();
        this.context = context;
    }

    @Override
    @TruffleBoundary
    public Object call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        checkArgumentLength(arguments, 3);

        String namespaceName = expectString(arguments[0], "argument 1 of bindall must be namespace string");
        String libraryFile = expectString(arguments[1], "argument 2 of bindall must be library file name (.so)");
        String nidlFile = expectString(arguments[2], "argument 3 of bindkall must bei NIDL file name (.nidl)");

        String[] namespaceComponents = splitNamespaceString(namespaceName);
        ArrayList<Binding> bindings = NIDLParser.parseNIDLFile(nidlFile);
        NamespaceTriple namespaceTriple = getOrCreateNamespace(namespaceComponents);
        bindings.forEach(binding -> binding.setLibraryFileName(libraryFile));

        // check binding type
        boolean kernelBindingPresent = false;
        boolean functionBindingPresent = false;
        for (Binding binding : bindings) {
            kernelBindingPresent |= binding instanceof KernelBinding;
            functionBindingPresent |= binding instanceof FunctionBinding;
        }
        if (!kernelBindingPresent && !functionBindingPresent) {
            // nothing to do
            return namespaceTriple.parentNamespace;
        }
        if (kernelBindingPresent && functionBindingPresent) {
            // NIDL file cannot contain both kernel and host function bindings
            throw new GrCUDAException("kernel and host function binding specified, can either bind kernel or host function");
        }
        if (kernelBindingPresent) {
            bindings.forEach(binding -> namespaceTriple.leafNamespace.addKernel(new LazyKernel((KernelBinding) binding, grCUDAExecutionContext)));
        }
        if (functionBindingPresent) {
            bindings.forEach(binding -> namespaceTriple.leafNamespace.addFunction(new HostFunction((FunctionBinding) binding, grCUDAExecutionContext.getCudaRuntime())));
        }

        if (namespaceTriple.childNamespace == null) {
            return namespaceTriple.parentNamespace;
        } else {
            // only add child namespace if all preceding operations were completed successfully
            namespaceTriple.parentNamespace.addNamespace(namespaceTriple.childNamespace);
            return namespaceTriple.leafNamespace;
        }
    }

    private NamespaceTriple getOrCreateNamespace(String[] namespaceComponents) {
        Namespace namespace = context.getRootNamespace();
        int pos = 0;
        for (; pos < namespaceComponents.length; pos++) {
            Optional<Object> maybeNamespaceObject = namespace.lookup(namespaceComponents[pos]);
            if (!maybeNamespaceObject.isPresent()) {
                // components[0] ... components[pos-1] are existing namespaces
                // but component[pos] does not exist.
                break;
            }
            Object o = maybeNamespaceObject.get();
            if (!(o instanceof Namespace)) {
                throw new GrCUDAException("Identifier " + namespaceComponents[pos] + " does not refer to a namespace");
            }
            namespace = (Namespace) o;
        }
        Namespace parentNamespace = namespace;
        Namespace childNamespace = null;
        while (pos < namespaceComponents.length) {
            Namespace ns = new Namespace(namespaceComponents[pos]);
            if (childNamespace == null) {
                childNamespace = ns;
            } else {
                namespace.addNamespace(ns);
            }
            namespace = ns;
            pos += 1;
        }
        return new NamespaceTriple(parentNamespace, childNamespace, namespace);
    }

    private final class NamespaceTriple {
        private final Namespace parentNamespace;
        private final Namespace childNamespace;
        private final Namespace leafNamespace;

        private NamespaceTriple(Namespace parentNs, Namespace childNs, Namespace leafNs) {
            this.parentNamespace = parentNs;
            this.childNamespace = childNs;
            this.leafNamespace = leafNs;
        }
    }

    private static String[] splitNamespaceString(String namespaceName) {
        String[] names = namespaceName.trim().split("::");
        if (names.length == 0) {
            throw new GrCUDAException("invalid namespace name");
        }
        // check whether namespace is composed of valid identifiers
        Pattern identifier = Pattern.compile("[a-zA-Z_]\\w*");
        for (String name : names) {
            if (!identifier.matcher(name).matches()) {
                throw new GrCUDAException("\"" + name + "\" is not a name in \"" + namespaceName + "\"");
            }
        }
        return names;
    }
}
