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

float R = 0.08;
float V = 0.3;
float T = 1.0;
float K = 60.0;

__device__ inline float cndGPU(float d) {
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

extern "C" __global__ void bs(const float *x, float *y, int N, float R, float V, float T, float K) {

    float sqrtT = __fdividef(1.0F, rsqrtf(T));
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float expRT;
        float d1, d2, CNDD1, CNDD2;
        d1 = __fdividef(__logf(x[i] / K) + (R + 0.5f * V * V) * T, V * sqrtT);
        d2 = d1 - V * sqrtT;

        CNDD1 = cndGPU(d1);
        CNDD2 = cndGPU(d2);

        //Calculate Call and Put simultaneously
        expRT = __expf(-R * T);
        y[i] = x[i] * CNDD1 - K * expRT * CNDD2;
    }
}

/////////////////////////////
/////////////////////////////

void init(float **y, float** x, int N, int K) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            x[i][j] = 60 - 0.5 + (float) rand() / RAND_MAX;
        }
    }
}

void reset(float *x, float** x_d, int N, int K, cudaStream_t *s) {
    // Copy just the first vector;
    cudaMemcpyAsync(x_d[0], x, sizeof(float) * N, cudaMemcpyHostToDevice, s[0]);
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

    int M = 10;

    if (debug) {
        std::cout << "running b5 default" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();
    
    float **x = (float **) malloc(sizeof(float*) * M);
    float **x_d = (float **) malloc(sizeof(float*) * M);
    float **y = (float **) malloc(sizeof(float*) * M);
    float **y_d = (float **) malloc(sizeof(float*) * M);
    // float *tmp_x = (float *) malloc(sizeof(float) * N);
    // cudaHostRegister(tmp_x, sizeof(float) * N, 0);

    for (int i = 0; i < M; i++) {
        x[i] = (float *) malloc(sizeof(float) * N);
        y[i] = (float *) malloc(sizeof(float) * N);
        cudaMalloc(&x_d[i], sizeof(float) * N);
        cudaMalloc(&y_d[i], sizeof(float) * N);
        cudaHostRegister(x[i], sizeof(float) * N, 0);
    }
    if (debug && err) std::cout << err << std::endl;
    
    // Create streams;
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * M);
    for (int i = 0; i < M; i++) {
        err = cudaStreamCreate(&s[i]);
    }
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;
    init(y, x, N, M);

    if (debug) std::cout << "x[0][0]=" << x[0][0] << std::endl;

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        // reset(x[0], x_d, N, M, s);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();

        for (int j = 0; j < M; j++) {
            if (j < M - 1) cudaMemcpyAsync(x_d[j + 1], x[j + 1], sizeof(float) * N, cudaMemcpyHostToDevice, s[j + 1]);
            bs<<<num_blocks, block_size, 0, s[j]>>>(x_d[j], y_d[j], N, R, V, T, K);
            if (j > 0) cudaMemcpyAsync(y[j - 1], y_d[j - 1], sizeof(float) * N, cudaMemcpyDeviceToHost, s[j - 1]);
        }

        // Copy last piece;
        cudaMemcpyAsync(y[M - 1], y_d[M - 1], sizeof(float) * N, cudaMemcpyDeviceToHost, s[M - 1]);

        for (int j = 0; j < M; j++) {
            err = cudaStreamSynchronize(s[j]);
        }

        if (debug && err) std::cout << err << std::endl;

        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        if (i >= skip_iterations) tot += tmp;

        if (debug) {
            std::cout << "  gpu result=[";
            for (int j = 0; j < M; j++) {
                std::cout << y[j][0] << ", ";
            } 
            std::cout << ", ...]; time=" << (float) tmp / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << y[0][0] << "," << (float) (reset_time + tmp) / 1e6 << "," << (float) reset_time / 1e6 << "," << (float) tmp / 1e6 << std::endl;
        }
    }

    // Print;
	cudaDeviceSynchronize();
    
    if (debug) std::cout << "\nmean exec time=" << (float) tot / (1000 * (num_executions - skip_iterations)) << " ms" << std::endl;
}
