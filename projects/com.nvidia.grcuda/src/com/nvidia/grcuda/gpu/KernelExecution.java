package com.nvidia.grcuda.gpu;

import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * Class used to track the single execution of a {@link ConfiguredKernel}.
 * The execution will be provided to the {@link GrCUDAExecutionContext} and scheduled accordingly.
 */
public class KernelExecution {

    private final Kernel kernel;
    private final ConfiguredKernel configuredKernel;
    private final KernelConfig config;
    private final KernelArguments args;

    public KernelExecution(ConfiguredKernel configuredKernel, KernelArguments args) {
        this.configuredKernel = configuredKernel;
        this.kernel = configuredKernel.getKernel();
        this.config = configuredKernel.getConfig();
        this.args = args;
    }

    public void execute() {
        kernel.getCudaRuntime().getExecutionContext().registerExecution(this);
        kernel.getCudaRuntime().cuLaunchKernel(kernel, config, args);
    }

    @Override
    public String toString() {
        return "KernelExecution(" + configuredKernel.toString() + "; args=[" +
                Arrays.stream(args.getOriginalArgs()).map(a -> Integer.toString(System.identityHashCode(a))).collect(Collectors.joining(", ")) +
                "])";
    }
}
