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
#include <set>

#include "benchmark.cuh"

typedef struct callBackData {
    float *n1;
    float *n2;
    int *r1;
    int *r2;
} callBackData_t;

class Benchmark7 : public Benchmark {
   public:
    Benchmark7(Options &options) : Benchmark(options) {}
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
    int degree = 3;
    int iterations = 5;
    int nnz;

    int *ptr, *idx, *val, *ptr2, *idx2, *val2, *rowCounter1, *rowCounter2, *x, *y, *v;
    int *ptr_tmp, *idx_tmp, *val_tmp, *ptr2_tmp, *idx2_tmp, *val2_tmp;
    float *auth1, *auth2, *hub1, *hub2, *auth_norm, *hub_norm;

    cudaStream_t s1, s2;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7;
    cudaGraphNode_t host_node;
    callBackData_t callback_data;
    cudaHostNodeParams host_params;
    cudaKernelNodeParams kernel_1_params;
    cudaKernelNodeParams kernel_2_params;
    cudaKernelNodeParams kernel_3_params;
    cudaKernelNodeParams kernel_4_params;
    cudaKernelNodeParams kernel_5_params;
    cudaKernelNodeParams kernel_6_params;
    cudaKernelNodeParams kernel_7_params;

    inline void random_coo(int *x, int *y, int *val, int N, int degree) {
        for (int i = 0; i < N; i++) {
            std::set<int> edges;
            while (edges.size() < degree) {
                edges.insert(rand() % N);
            }
            int j = 0;
            for (auto iter = edges.begin(); iter != edges.end(); iter++, j++) {
                x[i * degree + j] = i;
                y[i * degree + j] = *iter;
                val[i * degree + j] = 1;
            }
        }
    }
};
