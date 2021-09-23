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
package com.nvidia.grcuda.runtime.computation;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.ConfiguredKernel;
import com.nvidia.grcuda.runtime.Kernel;
import com.nvidia.grcuda.runtime.KernelArguments;
import com.nvidia.grcuda.runtime.KernelConfig;
import com.nvidia.grcuda.runtime.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.stream.DefaultStream;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Class used to track the single execution of a {@link ConfiguredKernel}.
 * The execution will be provided to the {@link GrCUDAExecutionContext} and scheduled accordingly.
 */
public class KernelExecution extends GrCUDAComputationalElement {

    private final Kernel kernel;
    private final ConfiguredKernel configuredKernel;
    private final KernelConfig config;
    private final KernelArguments args;

    public KernelExecution(ConfiguredKernel configuredKernel, KernelArguments args) {
        super(
            configuredKernel.getKernel().getGrCUDAExecutionContext(),
            new KernelExecutionInitializer(configuredKernel.getKernel(), args)
        );
        this.configuredKernel = configuredKernel;
        this.kernel = configuredKernel.getKernel();
        this.config = configuredKernel.getConfig();
        this.args = args;
    }

    @Override
    public Object execute() {
        grCUDAExecutionContext.getCudaRuntime().cuLaunchKernel(kernel, config, args, this.getStream());
        return NoneValue.get();
    }

    public ConfiguredKernel getConfiguredKernel() {
        return configuredKernel;
    }

    public KernelConfig getConfig() {
        return config;
    }

    public KernelArguments getArgs() {
        return args;
    }

    /**
     * Setting the stream must be done inside the {@link KernelConfig};
     * @param stream the stream where this computation will be executed
     */
    @Override
    public void setStream(CUDAStream stream) {
        // Make sure that the internal reference is consistent;
        super.setStream(stream);
    }

    /**
     * Retrieve the stream stored in the {@link KernelConfig} if it has been manually specified by the user,
     * otherwise return the one automatically chosen by the {@link com.nvidia.grcuda.runtime.stream.GrCUDAStreamManager};
     * @return the stream where this computation will be executed
     */
    @Override
    public CUDAStream getStream() {
        return config.useCustomStream() ? config.getStream() : super.getStream();
    }

    @Override
    public boolean useManuallySpecifiedStream() { return config.useCustomStream(); }

    @Override
    public boolean canUseStream() { return true; }

    @Override
    public void associateArraysToStreamImpl() {
        for (ComputationArgumentWithValue a : args.getKernelArgumentWithValues()) {
            if (a.getArgumentValue() instanceof AbstractArray) {
                AbstractArray array = (AbstractArray) a.getArgumentValue();
                if (getDependencyComputation().streamResetAttachFilter(a)) {
                    // If the array was attached to a stream, and now it is a const parameter, reset its visibility to the default stream;
                    if (!array.getStreamMapping().isDefaultStream()) {
                        grCUDAExecutionContext.getCudaRuntime().cudaStreamAttachMemAsync(DefaultStream.get(), array);
                    }
                } else if (!array.getStreamMapping().equals(this.getStream())) {
                    // Attach the array to the stream if the array isn't already attached to this stream;
                    grCUDAExecutionContext.getCudaRuntime().cudaStreamAttachMemAsync(this.getStream(), array);
                }
            }
        }
    }

    @Override
    public String toString() {
//        return "KernelExecution(" + configuredKernel.toString() + "; args=[" +
//                Arrays.stream(args.getOriginalArgs()).map(a -> Integer.toString(System.identityHashCode(a))).collect(Collectors.joining(", ")) +
//                "]" + "; stream=" + this.getStream() + ")";
        String event = this.getEvent().isPresent() ? Long.toString(this.getEvent().get().getEventNumber()) : "NULL";
        return "kernel=" + kernel.getKernelName() + "; args=[" +
                Arrays.stream(args.getOriginalArgs()).map(a -> Integer.toString(System.identityHashCode(a))).collect(Collectors.joining(", ")) +
                "]" + "; stream=" + this.getStream().getStreamNumber() + "; event=" + event;
    }

    static class KernelExecutionInitializer implements InitializeDependencyList {
        private final Kernel kernel;
        private final KernelArguments args;

        KernelExecutionInitializer(Kernel kernel, KernelArguments args) {
            this.kernel = kernel;
            this.args = args;
        }

        @Override
        public List<ComputationArgumentWithValue> initialize() {
            // TODO: what about scalars? We cannot treat them in the same way, as they are copied and not referenced
            //   There should be a semantic to manually specify scalar dependencies? For now we have to skip them;
            return this.args.getKernelArgumentWithValues().stream()
                    .filter(ComputationArgument::isArray).collect(Collectors.toList());
        }
    }
}


