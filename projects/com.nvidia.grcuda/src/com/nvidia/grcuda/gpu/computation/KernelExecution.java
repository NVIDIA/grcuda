package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.ComputationArgumentWithValue;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.ComputationArgument;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.ConfiguredKernel;
import com.nvidia.grcuda.gpu.Kernel;
import com.nvidia.grcuda.gpu.KernelArguments;
import com.nvidia.grcuda.gpu.KernelConfig;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.DefaultStream;

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
     * otherwise return the one automatically chosen by the {@link com.nvidia.grcuda.gpu.stream.GrCUDAStreamManager};
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


