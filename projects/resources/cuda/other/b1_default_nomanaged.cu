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

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include "utils.hpp"
#include "options.hpp"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

extern "C" __global__ void square(const float* x, float* y, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // float tmp = x[i];
        // float sum = 0;
        // for (int j = 0; j < 4; j++) {
        //     sum += tmp + j;
        // }

        y[i] = x[i]; // tmp + tmp * tmp / 2 + tmp * tmp * tmp / 6;
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// __device__ float atomicAddDouble(float* address, float val) {
//     unsigned long long int* address_as_ull = (unsigned long long int*) address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed, __float_as_longlong(val + __longlong_as_float(assumed)));
//     } while (assumed != old);
//     return __longlong_as_float(old);
// }

__global__ void reduce(const float *x, const float *y, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] - y[i];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

/////////////////////////////
/////////////////////////////

void init(float *x, float *y, int N) {
    for (int i = 0; i < N; i++) {
        x[i] = 1.0 / (i + 1);
        y[i] = 2.0 / (i + 1);
    }
}

void reset(float *res, float *res_d) {
    res[0] = 0.0;
    cudaMemcpy(res_d, res, sizeof(float), cudaMemcpyHostToDevice);
}

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

    srand(time(0));

    Options options = Options(argc, argv);
	int debug = options.debug;

    int num_executions = options.num_iter;
    int N = options.N;

    int block_size = options.block_size_1d;
    int num_blocks = options.num_blocks;
    int skip_iterations = options.skip_iterations;
    int err = 0;

    if (debug) {
        std::cout << "running b1 dag" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();
    float *x_d, *y_d, *x1_d, *y1_d, *res_d, *x, *y, *res;
    x = (float*) malloc(sizeof(float) * N);
    y = (float*) malloc(sizeof(float) * N);
    res = (float*) malloc(sizeof(float));
    err = cudaMalloc(&x_d, sizeof(float) * N);
    err = cudaMalloc(&y_d, sizeof(float) * N);
    err = cudaMalloc(&x1_d, sizeof(float) * N);
    err = cudaMalloc(&y1_d, sizeof(float) * N);
    err = cudaMalloc(&res_d, sizeof(float));
    if (debug && err) std::cout << err << std::endl;

    cudaHostRegister(x, sizeof(float) * N, 0);
    cudaHostRegister(y, sizeof(float) * N, 0);
    cudaHostRegister(res, sizeof(float), 0);

    // Create streams;
    cudaStream_t s1, s2;
    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;
    init(x, y, N);

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset(res, res_d);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();

        cudaMemcpyAsync(x_d, x, sizeof(float) * N, cudaMemcpyHostToDevice, s1);
        cudaMemcpyAsync(y_d, y, sizeof(float) * N, cudaMemcpyHostToDevice, s2);

        square<<<num_blocks, block_size, 0, s1>>>(x_d, x1_d, N);
        square<<<num_blocks, block_size, 0, s2>>>(y_d, y1_d, N);

        // Stream 1 waits stream 2;
        cudaEvent_t e1;
        cudaEventCreate(&e1);
        cudaEventRecord(e1, s2);
        cudaStreamWaitEvent(s1, e1, 0);

        reduce<<<num_blocks, block_size, 0, s1>>>(x1_d, y1_d, res_d, N);       
        cudaMemcpyAsync(res, res_d, sizeof(float), cudaMemcpyDeviceToHost, s1);
        cudaStreamSynchronize(s1); 

        if (debug && err) std::cout << err << std::endl;

        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        if (i >= skip_iterations) tot += tmp;

        if (debug) {
            std::cout << "  gpu result=" << res[0] << "; time=" << (float) tmp / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << res[0] << "," << (float) (reset_time + tmp) / 1e6 << "," << (float) reset_time / 1e6 << "," << (float) tmp / 1e6 << std::endl;
        }
    }

    // Print;
	cudaDeviceSynchronize();
    
    if (debug) std::cout << "\nmean exec time=" << (float) tot / (1000 * (num_executions - skip_iterations)) << " ms" << std::endl;
}
