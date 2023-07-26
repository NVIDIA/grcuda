/*
 * Copyright (c) 2022 NECSTLab, Politecnico di Milano. All rights reserved.
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

package it.necst.grcuda.benchmark;

import com.fasterxml.jackson.annotation.JsonIgnore;

/**
 * This class will be passed to initialize the configuration of a benchmark.
 */
public class BenchmarkConfig {
    /**
     * Default parameters
     */
    public String benchmarkName = "";
    public String setupId = "";
    public int totIter;
    public int currentIter;
    public int randomSeed = 42;
    public int size;
    public int blockSize1D = 32;
    public int blockSize2D = 8;
    boolean timePhases = false;
    public int numBlocks = 8;
    public boolean randomInit = false;
    public boolean reInit = false;
    public boolean reAlloc = false;
    public boolean cpuValidate = false;
    // GrCUDA context settings
    public String executionPolicy;
    public boolean inputPrefetch;
    public String retrieveNewStreamPolicy;
    public String retrieveParentStreamPolicy;
    public String dependencyPolicy;
    public String deviceSelectionPolicy;
    public boolean forceStreamAttach;
    public boolean enableComputationTimers;
    public int numGpus;
    public String memAdvisePolicy;
    @JsonIgnore public String bandwidthMatrix;
    // Debug parameters
    public boolean debug;
    public boolean nvprof_profile;
    public String gpuModel;
    @JsonIgnore public String results_path;

    @Override
    public String toString() {
        return "BenchmarkConfig{" +
                "benchmarkName='" + benchmarkName + '\'' +
                ", setupId='" + setupId + '\'' +
                ", totIter=" + totIter +
                ", currentIter=" + currentIter +
                ", randomSeed=" + randomSeed +
                ", size=" + size +
                ", blockSize1D=" + blockSize1D +
                ", blockSize2D=" + blockSize2D +
                ", timePhases=" + timePhases +
                ", numBlocks=" + numBlocks +
                ", randomInit=" + randomInit +
                ", reInit=" + reInit +
                ", reAlloc=" +reAlloc+
                ", cpuValidate=" + cpuValidate +
                ", executionPolicy='" + executionPolicy + '\'' +
                ", inputPrefetch=" + inputPrefetch +
                ", retrieveNewStreamPolicy='" + retrieveNewStreamPolicy + '\'' +
                ", retrieveParentStreamPolicy='" + retrieveParentStreamPolicy + '\'' +
                ", dependencyPolicy='" + dependencyPolicy + '\'' +
                ", deviceSelectionPolicy='" + deviceSelectionPolicy + '\'' +
                ", forceStreamAttach=" + forceStreamAttach +
                ", enableComputationTimers=" + enableComputationTimers +
                ", numGpus=" + numGpus +
                ", memAdvisePolicy='" + memAdvisePolicy + '\'' +
                ", bandwidthMatrix='" + bandwidthMatrix + '\'' +
                '}';
    }
}