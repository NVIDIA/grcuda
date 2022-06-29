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

#include "b1.cuh"

#define P 16

//////////////////////////////
//////////////////////////////

__global__ void square_m(const float *x, float *y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] * x[i];  
    }
}

__inline__ __device__ float warp_reduce_m(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduce_m(const float *x, const float *y, float *z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] - y[i];
    }
    sum = warp_reduce_m(sum);                    // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum);                     // The first thread in the warp updates the output;
}

//////////////////////////////
//////////////////////////////

void Benchmark1M::alloc() {

    S = (N + P - 1) / P;

    x = (float**) malloc(sizeof(float*) * P);
    y = (float**) malloc(sizeof(float*) * P);
    x1 = (float**) malloc(sizeof(float*) * P);
    y1 = (float**) malloc(sizeof(float*) * P);
    res = (float**) malloc(sizeof(float*) * P);
    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&x[i], sizeof(float) * S);
        err = cudaMallocManaged(&y[i], sizeof(float) * S);
        err = cudaMallocManaged(&x1[i], sizeof(float) * S);
        err = cudaMallocManaged(&y1[i], sizeof(float) * S);
        err = cudaMallocManaged(&res[i], sizeof(float));
    }
    
    // Create 2P streams;
    s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * 2 * P);
    for (int i = 0; i < P; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        err = cudaStreamCreate(&s[i]);
        err = cudaStreamCreate(&s[i + P]);
    }
}

void Benchmark1M::init() {
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < S; j++) {
            int index = i * S + j;
            if (index < N) {
                x[i][j] = 1.0 / (index + 1);
                y[i][j] = 2.0 / (index + 1);
            }
        }
    }
}

void Benchmark1M::reset() {
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < S; j++) {
            int index = i * S + j;
            if (index < N) {
                x[i][j] = 1.0 / (index + 1);
                y[i][j] = 2.0 / (index + 1);
            }
        }
        res[i][0] = 0.0;
    }
    res_tot = 0.0;
}

void Benchmark1M::execute_sync(int iter) {
    for (int i = 0; i < P; i++) {
        if (do_prefetch && pascalGpu) {
            cudaMemPrefetchAsync(x[i], sizeof(float) * S, 0, 0);
            cudaMemPrefetchAsync(x1[i], sizeof(float) * S, 0, 0);
            cudaMemPrefetchAsync(y[i], sizeof(float) * S, 0, 0);
            cudaMemPrefetchAsync(y1[i], sizeof(float) * S, 0, 0);
        }
        square_m<<<num_blocks, block_size_1d>>>(x[i], x1[i], S);
        err = cudaDeviceSynchronize();
        square_m<<<num_blocks, block_size_1d>>>(y[i], y1[i], S);
        err = cudaDeviceSynchronize();
        reduce_m<<<num_blocks, block_size_1d>>>(x1[i], y1[i], res[i], S);
        err = cudaDeviceSynchronize();
    }
    for (int i = 0; i < P; i++) {
        res_tot += res[i][0];
    }
}

void Benchmark1M::execute_async(int iter) {
    for (int i = 0; i < P; i++) {
        int gpu = select_gpu(i, max_devices);
        cudaSetDevice(gpu);
        if (!pascalGpu || stream_attach) {
            cudaStreamAttachMemAsync(s[i], x[i], sizeof(float) * S);
            cudaStreamAttachMemAsync(s[i], x1[i], sizeof(float) * S);
            cudaStreamAttachMemAsync(s[i + P], y[i], sizeof(float) * S);
            cudaStreamAttachMemAsync(s[i + P], y1[i], sizeof(float) * S);
        }
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(x[i], sizeof(float) * S, gpu, s[i]);
            cudaMemPrefetchAsync(x1[i], sizeof(float) * S, gpu, s[i]);
            cudaMemPrefetchAsync(y[i], sizeof(float) * S, gpu, s[i + P]);
            cudaMemPrefetchAsync(y1[i], sizeof(float) * S, gpu, s[i + P]);
        }

        square_m<<<num_blocks, block_size_1d, 0, s[i]>>>(x[i], x1[i], S);
        square_m<<<num_blocks, block_size_1d, 0, s[i + P]>>>(y[i], y1[i], S);

        // Stream 1 waits stream 2;
        cudaEvent_t e1;
        cudaEventCreate(&e1);
        cudaEventRecord(e1, s[i + P]);
        cudaStreamWaitEvent(s[i], e1, 0);

        reduce_m<<<num_blocks, block_size_1d, 0, s[i]>>>(x1[i], y1[i], res[i], S);
    }
    for (int i = 0; i < P; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        cudaStreamSynchronize(s[i]);
        res_tot += res[i][0];        
    }
}

std::string Benchmark1M::print_result(bool short_form) {
    return std::to_string(res_tot);
}