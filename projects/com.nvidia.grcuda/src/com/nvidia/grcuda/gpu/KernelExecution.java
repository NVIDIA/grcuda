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

    // TODO: this should be moved somewhere else, like in a higher level interface that implements
    //  this function with different strategies

    /**
     * Computes if the "other" KernelExecution has dependencies w.r.t. this kernel,
     * such as requiring as input a value computed by this kernel.
     * @param other kernel for which we want to check dependencies, w.r.t. this kernel
     * @return true IFF at least one dependency was found
     */
    public boolean hasDependency(KernelExecution other) {
        return true;
    }

    @Override
    public String toString() {
        return "KernelExecution(" + configuredKernel.toString() + "; args=[" +
                Arrays.stream(args.getOriginalArgs()).map(a -> Integer.toString(System.identityHashCode(a))).collect(Collectors.joining(", ")) +
                "])";
    }
}
