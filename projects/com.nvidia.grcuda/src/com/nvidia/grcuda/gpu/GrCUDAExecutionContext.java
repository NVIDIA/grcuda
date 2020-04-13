package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.array.AbstractArray;

import java.util.HashSet;
import java.util.Set;

/**
 * Class used to monitor the state of GrCUDA execution, keep track of memory allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public class GrCUDAExecutionContext {

    /**
     * Set that contains all the arrays allocated so far.
     */
    final private Set<AbstractArray> arraySet = new HashSet<>();

    /**
     * Set that contains all the CUDA kernels declared so far.
     */
    final private Set<Kernel> kernelSet = new HashSet<>();

    /**
     * Set that contains all the CUDA kernels execution so far.
     * TODO: this should not be a set, but a DAG that can be used to handle dependencies
     */
    final private Set<KernelExecution> kernelExecSet = new HashSet<>();

    public GrCUDAExecutionContext() {

    }

    public void registerArray(AbstractArray array) {
        arraySet.add(array);
        System.out.println("-- added array to context: " + System.identityHashCode(array) + "; " + array.toString());
    }

    public void registerKernel(Kernel kernel) {
        kernelSet.add(kernel);
        System.out.println("-- added kernel to context: " + System.identityHashCode(kernel) + "; " + kernel.toString());
    }

    public void registerExecution(KernelExecution kernel) {
        kernelExecSet.add(kernel);
        System.out.println("-- executing kernel: " + System.identityHashCode(kernel) + "; " + kernel.toString());
    }
}
