// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include "benchmark.cuh"

class Benchmark10 : public Benchmark {
   public:
    Benchmark10(Options &options) : Benchmark(options) {}
    void alloc();
    void init();
    void reset();
    void execute_sync(int iter);
    void execute_async(int iter);
    void execute_cudagraph(int iter);
    void execute_cudagraph_manual(int iter);
    void execute_cudagraph_single(int iter);
    std::string print_result(bool short_form = false);

   private:
    int K = 3;
    int channels = 1;
    int stride = 2;
    int kn1 = 8;
    int kn2 = 16;
    int pooling_diameter = 5;

    float *x, *x1, *x2, *x3, *y, *y1, *y2, *y3, *kernel_1, *kernel_2, *kernel_3, *kernel_4, *z, *dense_weights, *res;
    float *x11, *y11;
    float *x_cpu;
    float *y_cpu;
    int x_len;
    int x1_len;
    int pooled_len;
    int x2_len;
    int x3_len;
    int k1_len, k2_len, z_len;

    cudaStream_t s1, s2;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8;
    cudaKernelNodeParams kernel_1_params;
    cudaKernelNodeParams kernel_2_params;
    cudaKernelNodeParams kernel_3_params;
    cudaKernelNodeParams kernel_4_params;
    cudaKernelNodeParams kernel_5_params;
    cudaKernelNodeParams kernel_6_params;
    cudaKernelNodeParams kernel_7_params;
    cudaKernelNodeParams kernel_8_params;
};
