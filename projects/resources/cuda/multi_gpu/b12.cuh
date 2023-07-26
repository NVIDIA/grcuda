// Copyright (c) 2021, NECSTLab, Politecnico di Milano. All rights reserved.

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
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cassert>
#include <thread>
#include <ostream>
#include <fstream>
#include <cstring>
#include <sstream>

#include "../benchmark.cuh"
#include "../mmio.hpp"

using f32 = float;
using u32 = unsigned;
using f64 = double;
using u64 = long;
using i32 = int;

struct coo_matrix_t {
    friend std::ostream &operator<<(std::ostream &os, const coo_matrix_t &matrix);

    i32 *x;
    i32 *y;
    float *val;
    i32 begin;
    i32 end;
    i32 N;
    i32 nnz;

};



#define DOT_PRODUCT_NUM_BLOCKS 32
#define DOT_PRODUCT_THREADS_PER_BLOCK 64

class Benchmark12M : public Benchmark {
public:
    Benchmark12M(Options &options) : Benchmark(options) {
        // This test does not run on pascal gpus due to how Managed memory is handled

        this->block_size = this->block_size_1d * this->block_size_2d;
        this->num_partitions = options.max_devices;


        cudaGetDeviceCount(&this->num_devices);
        //this->num_devices = std::min(this->num_devices, this->num_partitions);
        assert(this->num_devices > 0);


    }
    void alloc();
    void init();
    void reset();
    void execute_sync(i32);
    void execute_async(i32);
    void execute_cudagraph(i32);
    void execute_cudagraph_manual(i32);
    void execute_cudagraph_single(i32);
    void load_matrix(bool);
    std::string print_result(bool);


private:

    unsigned num_eigencomponents = 8;
    i32 num_partitions = -1;
    i32 num_devices = -1;
    std::string matrix_path = "../datasets/333SP.mtx";
    bool reorthogonalize = false;
    i32 block_size;
    coo_matrix_t matrix;
    std::vector<coo_matrix_t*> coo_partitions;
    f32 *alpha, *beta;
    std::vector<float*> vec_in, spmv_vec_out, intermediate_dot_product_values,  vec_next, lanczos_vectors, normalized_out;
    float *alpha_intermediate, *beta_intermediate;
    std::vector<cudaStream_t> streams;
    std::vector<i32> offsets;

    std::vector<f32> tridiagonal_matrix;

    void alloc_coo_partitions();
    void alloc_vectors();
    void create_random_matrix(bool);
    void execute(i32);
    void sync_all();
    void create_streams();
    coo_matrix_t *assign_partition(unsigned, unsigned, unsigned);

    template<typename Function>
    void launch_multi_kernel(Function);

    static constexpr u32 RANDOM_MATRIX_NUM_ROWS = 1000000;
    static constexpr u32 RANDOM_MATRIX_AVG_NNZ_PER_ROW = 100;

};

