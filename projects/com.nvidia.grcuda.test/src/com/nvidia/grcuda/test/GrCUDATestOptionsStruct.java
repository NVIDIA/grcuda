package com.nvidia.grcuda.test;

public class GrCUDATestOptionsStruct {
    public final String policy;
    public final boolean inputPrefetch;
    public final String retrieveNewStreamPolicy;
    public final String retrieveParentStreamPolicy;
    public final String dependencyPolicy;
    public final boolean forceStreamAttach;

    /**
     * A simple struct that holds a combination of GrCUDA options, extracted from the output of {@link GrCUDATestUtil#getAllOptionCombinations}
     */
    public GrCUDATestOptionsStruct(String policy, boolean inputPrefetch, String retrieveNewStreamPolicy, String retrieveParentStreamPolicy, String dependencyPolicy, boolean forceStreamAttach) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
        this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
        this.retrieveParentStreamPolicy = retrieveParentStreamPolicy;
        this.dependencyPolicy = dependencyPolicy;
        this.forceStreamAttach = forceStreamAttach;
    }
}
