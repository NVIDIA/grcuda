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
#include <chrono>
#include <iostream>
#include <string>

#include "options.hpp"
#include "utils.hpp"

struct Benchmark {
   public:
    virtual void alloc() = 0;
    virtual void init() = 0;
    virtual void reset() = 0;
    virtual void execute_async(int iter);
    virtual void execute_sync(int iter);
    virtual void execute_cudagraph(int iter);
    virtual void execute_cudagraph_manual(int iter);
    virtual void execute_cudagraph_single(int iter);
    virtual std::string print_result(bool short_form = false) = 0;
    virtual void cpu_validation(int iter);
    void run();
    int add_node(void **paramarray, cudaKernelNodeParams &param, void *func, dim3 gridsize, dim3 threads, cudaGraph_t &g, cudaGraphNode_t *n, std::vector<cudaGraphNode_t> &dependencies, int shared_memory = 0);
    int select_gpu(int i, int max_devices);

    Benchmark(Options &options) : debug(options.debug),
                                  num_executions(options.num_iter),
                                  N(options.N),
                                  block_size_1d(options.block_size_1d),
                                  block_size_2d(options.block_size_2d),
                                  num_blocks(options.num_blocks),
                                  skip_iterations(options.skip_iterations),
                                  do_prefetch(options.prefetch),
                                  stream_attach(options.stream_attach),
                                  policy(options.policy_choice),
                                  benchmark_name(options.benchmark_choice),
                                  max_devices(options.max_devices),
                                  nvprof(options.nvprof),
                                  num_partitions(options.num_partitions) {
        cudaDeviceGetAttribute(&pascalGpu, cudaDeviceAttr::cudaDevAttrConcurrentManagedAccess, 0);
        if (debug) {
            std::cout << "------------------------------" << std::endl;
            std::cout << "- running " << options.benchmark_map[benchmark_name] << std::endl;
            std::cout << "- num executions=" << num_executions << std::endl;
            std::cout << "- iterations to skip=" << skip_iterations << std::endl;
            std::cout << "- N=" << N << std::endl;
            std::cout << "- policy=" << options.policy_map[policy] << std::endl;
            std::cout << "- block size 1d=" << block_size_1d << std::endl;
            std::cout << "- block size 2d (where applicable)=" << block_size_2d << std::endl;
            std::cout << "- num blocks=" << num_blocks << std::endl;
            std::cout << "- max devices (where applicable)=" << max_devices << std::endl;
            std::cout << "- use nvprof=" << nvprof << std::endl;
            std::cout << "- num of partitions (where applicable)=" << num_partitions << std::endl;
            std::cout << "------------------------------" << std::endl;
        }
    }

    virtual ~Benchmark(){};

   protected:
    int debug = DEBUG;
    int num_executions = NUM_ITER;
    int N = 0;
    int block_size_1d = DEFAULT_BLOCK_SIZE_1D;
    int block_size_2d = DEFAULT_BLOCK_SIZE_2D;
    int num_blocks = DEFAULT_NUM_BLOCKS;
    int skip_iterations = 0;
    bool do_prefetch = DEFAULT_PREFETCH;
    bool stream_attach = DEFAULT_STREAM_ATTACH;
    int max_devices = DEFAULT_MAX_DEVICES;
    int pascalGpu = 0;
    bool nvprof = DEFAULT_NVPROF;
    int num_partitions = DEFAULT_NUM_PARTITIONS;
    Policy policy;
    BenchmarkEnum benchmark_name;
    int err = 0;
};
