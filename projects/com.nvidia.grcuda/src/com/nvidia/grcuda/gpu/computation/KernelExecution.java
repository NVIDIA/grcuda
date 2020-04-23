package com.nvidia.grcuda.gpu.computation;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.array.DeviceArray;
import com.nvidia.grcuda.gpu.ConfiguredKernel;
import com.nvidia.grcuda.gpu.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.Kernel;
import com.nvidia.grcuda.gpu.KernelArguments;
import com.nvidia.grcuda.gpu.KernelConfig;
import com.nvidia.grcuda.gpu.stream.CUDAStream;

import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
        for (Object o : this.args.getOriginalArgs()) {
            // FIXME: this should also be done for other array types;
            if (o instanceof DeviceArray) {
                // Attach the array to the stream, synchronously;
                grCUDAExecutionContext.getCudaRuntime().cudaStreamAttachMem(this.getStream(), (DeviceArray) o);
            }
        }
    }

    @Override
    public String toString() {
        return "KernelExecution(" + configuredKernel.toString() + "; args=[" +
                Arrays.stream(args.getOriginalArgs()).map(a -> Integer.toString(System.identityHashCode(a))).collect(Collectors.joining(", ")) +
                "]" + "; stream=" + this.getStream() + ")";
    }

    static class KernelExecutionInitializer implements InitializeArgumentSet {
        private final Kernel kernel;
        private final KernelArguments args;

        KernelExecutionInitializer(Kernel kernel, KernelArguments args) {
            this.kernel = kernel;
            this.args = args;
        }

        @Override
        public Set<Object> initialize() {
            // TODO: what aboout scalars? We cannot treat them in the same way, as they are copied and not referenced
            //   There should be a semantic to manually specify scalar dependencies? For now we have to skip them;
            return IntStream.range(0, args.getOriginalArgs().length).filter(i ->
                    kernel.getArgsAreArrays().get(i)
            ).mapToObj(args::getOriginalArg).collect(Collectors.toSet());
        }
    }
}


