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

double R = 0.08;
double V = 0.3;
double T = 1.0;
double K = 60.0;

__device__ inline double
cndGPU(double d) {
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
bs(const double *x, double *y, int N, double R, double V, double T, double K) {
    double sqrtT = 1.0 / rsqrt(T);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        double expRT;
        double d1, d2, CNDD1, CNDD2;
        d1 = (log(x[i] / K) + (R + 0.5 * V * V) * T) / (V * sqrtT);
        d2 = d1 - V * sqrtT;

        CNDD1 = cndGPU(d1);
        CNDD2 = cndGPU(d2);

        // Calculate Call and Put simultaneously
        expRT = exp(-R * T);
        y[i] = x[i] * CNDD1 - K * expRT * CNDD2;
    }
}

void prefetch(double *x, double *y, cudaStream_t &s, int N) {
    int pascalGpu = 0;
    cudaDeviceGetAttribute(&pascalGpu, cudaDeviceAttr::cudaDevAttrConcurrentManagedAccess, 0);
    if (pascalGpu) {
        cudaMemPrefetchAsync(x, sizeof(double) * N, 0, s);
        cudaMemPrefetchAsync(y, sizeof(double) * N, 0, s);
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark5::alloc() {
    x = (double **)malloc(sizeof(double *) * M);
    y = (double **)malloc(sizeof(double *) * M);
    tmp_x = (double *)malloc(sizeof(double) * N);
    // cudaHostRegister(tmp_x, sizeof(double) * N, 0);

    for (int i = 0; i < M; i++) {
        cudaMallocManaged(&x[i], sizeof(double) * N);
        cudaMallocManaged(&y[i], sizeof(double) * N);
    }
}

void Benchmark5::init() {
    for (int j = 0; j < N; j++) {
        tmp_x[j] = 60 - 0.5 + (double)rand() / RAND_MAX;
        for (int i = 0; i < M; i++) {
            x[i][j] = tmp_x[j];
            // y[i][j] = 0;
        }
    }

    s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * M);
    for (int i = 0; i < M; i++) {
        err = cudaStreamCreate(&s[i]);
    }
}

void Benchmark5::reset() {
    for (int i = 0; i < M; i++) {
        // memcpy(x[i], y, sizeof(int) * N);
        // cudaMemcpy(x[i], y, sizeof(double) * N, cudaMemcpyDefault);

        // cudaMemcpyAsync(x[i], y, sizeof(int) * N, cudaMemcpyHostToDevice,
        // s[i]);
        for (int j = 0; j < N; j++) {
            x[i][j] = tmp_x[j];
        }
    }
    // cudaMemPrefetchAsync(x[0], sizeof(double) * N, 0, s[0]);
}

void Benchmark5::execute_sync(int iter) {
    for (int j = 0; j < M; j++) {
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(x[j], sizeof(double) * N, 0, 0);
            cudaMemPrefetchAsync(y[j], sizeof(double) * N, 0, 0);
        }
        bs<<<num_blocks, block_size_1d>>>(x[j], y[j], N, R, V, T, K);
        err = cudaDeviceSynchronize();
    }
}

void Benchmark5::execute_async(int iter) {
    for (int j = 0; j < M; j++) {
        if (!pascalGpu || stream_attach) {
            cudaStreamAttachMemAsync(s[j], x[j], sizeof(double) * N);
            cudaStreamAttachMemAsync(s[j], y[j], sizeof(double) * N);
        }
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(x[j], sizeof(double) * N, 0, s[j]);
            cudaMemPrefetchAsync(y[j], sizeof(double) * N, 0, s[j]);
        }
        // if (j > 0) cudaMemPrefetchAsync(y[j - 1], sizeof(double) * N, cudaCpuDeviceId, s[j - 1]);
        bs<<<num_blocks, block_size_1d, 0, s[j]>>>(x[j], y[j], N, R, V, T, K);
        // if (j < M - 1) cudaMemPrefetchAsync(x[j + 1], sizeof(double) * N, 0, s[j + 1]);
    }

    // Last tile;
    // cudaMemPrefetchAsync(y[M - 1], sizeof(double) * N, cudaCpuDeviceId, s[M - 1]);

    for (int j = 0; j < M; j++) {
        err = cudaStreamSynchronize(s[j]);
    }
}

void Benchmark5::execute_cudagraph(int iter) {
    if (iter == 0) {
        for (int j = 0; j < M; j++) {
            cudaStreamBeginCapture(s[j], cudaStreamCaptureModeGlobal);
            // prefetch(x[j], y[j], s[j], N);
            bs<<<num_blocks, block_size_1d, 0, s[j]>>>(x[j], y[j], N, R, V, T, K);
            cudaStreamEndCapture(s[j], &graphs[j]);
            cudaGraphInstantiate(&graphExec[j], graphs[j], NULL, NULL, 0);
        }
    }
    for (int j = 0; j < M; j++) {
        cudaGraphLaunch(graphExec[j], s[j]);
    }
    for (int j = 0; j < M; j++) {
        cudaStreamSynchronize(s[j]);
    }
}

void Benchmark5::execute_cudagraph_manual(int iter) {
    if (iter == 0) {
        cudaGraphCreate(&graphs[0], 0);
        for (int j = 0; j < M; j++) {
            void *kernel_args[7] = {(void *)&x[j], (void *)&y[j], &N, &R, &V, &T, &K};

            dim3 tb(block_size_1d);
            dim3 b_size(num_blocks);

            // bs<<<num_blocks, block_size_1d, 0, s[j]>>>(x[j], y[j], N, R, V, T, K);
            add_node(kernel_args, kernel_params[j], (void *)bs, b_size, tb, graphs[0], &kernels[j], nodeDependencies);
        }
        cudaGraphInstantiate(&graphExec[0], graphs[0], NULL, NULL, 0);
    }
    cudaGraphLaunch(graphExec[0], s[0]);
    err = cudaStreamSynchronize(s[0]);
}

void Benchmark5::execute_cudagraph_single(int iter) {
    if (iter == 0) {
        cudaStreamBeginCapture(s[0], cudaStreamCaptureModeGlobal);
        for (int j = 0; j < M; j++) {
            // prefetch(x[j], y[j], s[0], N);
            bs<<<num_blocks, block_size_1d, 0, s[0]>>>(x[j], y[j], N, R, V, T, K);
        }
        cudaStreamEndCapture(s[0], &graphs[0]);
        cudaGraphInstantiate(&graphExec[0], graphs[0], NULL, NULL, 0);
    }
    cudaGraphLaunch(graphExec[0], s[0]);
    cudaStreamSynchronize(s[0]);
}

std::string
Benchmark5::print_result(bool short_form) {
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
