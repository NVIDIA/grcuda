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

#include "b5.cuh"

//////////////////////////////
//////////////////////////////


__device__ inline double
cndGPU_m(double d) {
    const double A1 = 0.31938153f;
    const double A2 = -0.356563782f;
    const double A3 = 1.781477937f;
    const double A4 = -1.821255978f;
    const double A5 = 1.330274429f;
    const double RSQRT2PI = 0.39894228040143267793994605993438f;

    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double cnd = RSQRT2PI * exp(-0.5f * d * d) *
                 (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

extern "C" __global__ void
bs_m(const double *x, double *y, int N, double R, double V, double T, double K) {
    double sqrtT = 1.0 / rsqrt(T);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        double expRT;
        double d1, d2, CNDD1, CNDD2;
        d1 = (log(x[i] / K) + (R + 0.5 * V * V) * T) / (V * sqrtT);
        d2 = d1 - V * sqrtT;

        CNDD1 = cndGPU_m(d1);
        CNDD2 = cndGPU_m(d2);

        // Calculate Call and Put simultaneously
        expRT = exp(-R * T);
        y[i] = x[i] * CNDD1 - K * expRT * CNDD2;
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark5M::alloc() {
    x = (double**) malloc(sizeof(double *) * M);
    y = (double**) malloc(sizeof(double *) * M);
    tmp_x = (double*) malloc(sizeof(double) * N);

    for (int i = 0; i < M; i++) {
        cudaMallocManaged(&x[i], sizeof(double) * N);
        cudaMallocManaged(&y[i], sizeof(double) * N);
    }
}

void Benchmark5M::init() {
    for (int j = 0; j < N; j++) {
        tmp_x[j] = 60 - 0.5 + (double) rand() / RAND_MAX;
        for (int i = 0; i < M; i++) {
            x[i][j] = tmp_x[j];
        }
    }

    s = (cudaStream_t*) malloc(sizeof(cudaStream_t) * M);
    for (int i = 0; i < M; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        err = cudaStreamCreate(&s[i]);
    }
}

void Benchmark5M::reset() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            x[i][j] = tmp_x[j];
        }
    }
}

void Benchmark5M::execute_sync(int iter) {
    for (int j = 0; j < M; j++) {
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(x[j], sizeof(double) * N, 0, 0);
            cudaMemPrefetchAsync(y[j], sizeof(double) * N, 0, 0);
        }
        bs_m<<<num_blocks, block_size_1d>>>(x[j], y[j], N, R, V, T, K);
        err = cudaDeviceSynchronize();
    }
}

void Benchmark5M::execute_async(int iter) {
    for (int j = 0; j < M; j++) {
        int gpu = select_gpu(j, max_devices);
        cudaSetDevice(gpu);
        if (!pascalGpu || stream_attach) {
            cudaStreamAttachMemAsync(s[j], x[j], sizeof(double) * N);
            cudaStreamAttachMemAsync(s[j], y[j], sizeof(double) * N);
        }
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(x[j], sizeof(double) * N, gpu, s[j]);
            cudaMemPrefetchAsync(y[j], sizeof(double) * N, gpu, s[j]);
        }
        bs_m<<<num_blocks, block_size_1d, 0, s[j]>>>(x[j], y[j], N, R, V, T, K);
    }

    for (int j = 0; j < M; j++) {
        err = cudaStreamSynchronize(s[j]);
    }
}

std::string
Benchmark5M::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(y[0][0]);
    } else {
        std::string res = "[";
        for (int j = 0; j < M; j++) {
            res += std::to_string(y[j][0]) + ", ";
        }
        return res + ", ...]";
    }
}