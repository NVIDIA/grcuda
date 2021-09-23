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

#include "benchmark.cuh"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

int Benchmark::add_node(void **paramarray, cudaKernelNodeParams &param, void *func, dim3 gridsize, dim3 threads, cudaGraph_t &g, cudaGraphNode_t *n, std::vector<cudaGraphNode_t> &dependencies, int shared_memory) {
    param.func = func;
    param.blockDim = threads;
    param.gridDim = gridsize;
    param.kernelParams = paramarray;
    param.sharedMemBytes = shared_memory;
    param.extra = NULL;
    return cudaGraphAddKernelNode(n, g, dependencies.data(), dependencies.size(), &param);
}

void Benchmark::run() {
    auto start_tot = clock_type::now();
    auto start_tmp = clock_type::now();
    auto end_tmp = clock_type::now();

    // Allocation;
    start_tmp = clock_type::now();
    alloc();
    end_tmp = clock_type::now();
    if (debug && err) std::cout << "error=" << err << std::endl;
    if (debug) std::cout << "allocation time=" << chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count() / 1000 << " ms" << std::endl;

    // Initialization;
    start_tmp = clock_type::now();
    init();
    end_tmp = clock_type::now();
    if (debug && err) std::cout << "error=" << err << std::endl;
    if (debug) std::cout << "initialization time=" << chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;

    long tot_time = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;

        // Reset;
        start_tmp = clock_type::now();
        reset();
        end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float)reset_time / 1000 << " ms" << std::endl;

        // Execution;
        start_tmp = clock_type::now();
        switch (policy) {
            case Policy::Sync:
                execute_sync(i);
                break;
            case Policy::CudaGraph:
                execute_cudagraph(i);
                break;
            case Policy::CudaGraphAsync:
                execute_cudagraph_manual(i);
                break;
            case Policy::CudaGraphSingle:
                execute_cudagraph_single(i);
                break;
            default:
                execute_async(i);
        }
        if (debug && err) std::cout << "  error=" << err << std::endl;
        end_tmp = clock_type::now();
        auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();

        if (i >= skip_iterations)
            tot_time += exec_time;

        if (debug) {
            std::cout << "  result=" << print_result() << std::endl;
            std::cout << "  execution(" << i << ")=" << (float)exec_time / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << print_result(true) << "," << (float)(reset_time + exec_time) / 1e6 << "," << (float)reset_time / 1e6 << "," << (float)exec_time / 1e6 << std::endl;
        }
    }

    auto end_time = chrono::duration_cast<chrono::microseconds>(clock_type::now() - start_tot).count();
    if (debug) std::cout << "\ntotal execution time=" << end_time / 1e6 << " sec" << std::endl;
    if (debug) std::cout << "mean exec time=" << (float)tot_time / (1000 * (num_executions - skip_iterations)) << " ms" << std::endl;
}
