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

#include "b9.cuh"

//////////////////////////////
//////////////////////////////

#define BLOCK_SIZE_V100 64 // Just a recommendation of optimal block size for the V100;
#define P 16
#define ITER 50
#define EPS 1e-12

// Add a small epsilon to the main diagonal:
extern "C" __global__ void precondition(float *A, int n, int m, int offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
        A[i * n + i + offset] += EPS; 
    }
}

// z = x @ y;
extern "C" __global__ void matrix_vector_mult(const float* x, const float* y, float* z, int n, int m, int z_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[z_offset + i] = sum;
    }
}

// z := w + alpha * A @ y;
extern "C" __global__ void matrix_vector_mult_axpy(const float* x, const float* y, const float *w, const float alpha, float* z, int n, int m, int z_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[z_offset + i] = alpha * sum + w[z_offset + i];
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// z = <x, x>;
extern "C" __global__ void l2_norm(const float *x, float* z, int N, int offset) {
    int warp_size = 32;
    float sum = 0;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float x_tmp = x[i + offset];
        sum += x_tmp * x_tmp;
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

// z = <x, y>;
extern "C" __global__ void dot(const float *x, const float *y, float* z, int N, int offset) {
    int warp_size = 32;
    float sum = 0;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i + offset] * y[i + offset];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

// y = val + alpha * x;
extern "C" __global__ void saxpy(float* y, const float *val, const float *x, float alpha, int n, int offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i + offset] = val[i + offset] + alpha * x[i + offset];
    }
}

// Simply copy array x into y;
extern "C" __global__ void cpy(float *y, const float *x, int n, int offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i + offset] = x[i + offset];
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark9M::alloc() {
    S = (N + P - 1) / P;
    A = (float **) malloc(sizeof(float*) * P);

    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&A[i], sizeof(float) * S * N);
    }
    err = cudaMallocManaged(&x, sizeof(float) * N);
    err = cudaMallocManaged(&b, sizeof(float) * N);
    err = cudaMallocManaged(&p, sizeof(float) * N);
    err = cudaMallocManaged(&r, sizeof(float) * N);
    err = cudaMallocManaged(&y, sizeof(float) * N);

    err = cudaMallocManaged(&t1, sizeof(float));
    err = cudaMallocManaged(&t2, sizeof(float));
    // Other implementation;
    // t1 = (float **) malloc(sizeof(float*) * P);
    // t2 = (float **) malloc(sizeof(float*) * P);
    // for (int i = 0; i < P; i++) {
    //     err = cudaMallocManaged(&t1[i], sizeof(float));
    //     err = cudaMallocManaged(&t2[i], sizeof(float));
    // }

    // Create streams;
    cudaStream_t s1, s2;
    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);
    // Create P streams;
    s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * P);
    for (int i = 0; i < P; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        err = cudaStreamCreate(&s[i]);
    }
    cudaSetDevice(select_gpu(0, max_devices));
}

void Benchmark9M::init() {
    // Random symmetric invertible input matrix;
    float max = float(RAND_MAX);

    for (int i = 0; i < N; i++) {
        int p = i / S;
        for (int j = i; j < N; j++) {
            float val = (float(rand()) / max) * 2 - 1;
            A[p][(i % S) * N + j] = val;
            // A[j / S][(j % S) * N + i] = val;
        }
        // A[p][(i % S) * N + i] += 10e-12; 
    }

    // Random input b;
    for (int i = 0; i < N; i++) {
        b[i] = (float(rand()) / max) * 2 - 1;
    }
}

void Benchmark9M::reset() {
    // Default init of solution x;
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
    }
    // Reset norms;
    *t1 = 0.0;
    *t2 = 0.0;
    // Other implementation;
    // t1_tot = 0.0;
    // t2_tot = 0.0;
    // for (int i = 0; i < P; i++) {
    //     t1[i][0] = 0;
    //     t2[i][0] = 0;
    // }
}

void Benchmark9M::execute_sync(int iter) { 

    if (pascalGpu && do_prefetch) {
        for (int i = 0; i < P; i++) {
            cudaMemPrefetchAsync(A[i], sizeof(float) * S * N, 0);
        }
        cudaMemPrefetchAsync(x, sizeof(float) * N, 0);
        cudaMemPrefetchAsync(b, sizeof(float) * N, 0);
        cudaMemPrefetchAsync(r, sizeof(float) * N, 0);
        cudaMemPrefetchAsync(p, sizeof(float) * N, 0);
    }

    for (int i = 0; i < P; i++) {
        precondition<<<num_blocks, block_size_1d>>>(A[i], N, std::min(S, N - i * S), i * S);
        matrix_vector_mult_axpy<<<num_blocks, block_size_1d>>>(A[i], x, b, -1, r, std::min(S, N - i * S), N, i * S);
        cudaDeviceSynchronize();
    }
    cpy<<<num_blocks, block_size_1d>>>(p, r, N, 0);
    cudaDeviceSynchronize();
    l2_norm<<<num_blocks, block_size_1d>>>(r, t1, N, 0);
    cudaDeviceSynchronize();
    for (int iter = 0; iter < ITER; iter++) {
        for (int i = 0; i < P; i++) {
            matrix_vector_mult<<<num_blocks, block_size_1d>>>(A[i], p, y, std::min(S, N - i * S), N, i * S);
            cudaDeviceSynchronize();
        }
        dot<<<num_blocks, block_size_1d>>>(p, y, t2, N, 0);
        cudaDeviceSynchronize();
        float alpha = *t1 / *t2;
        float old_t1 = *t1;
        *t1 = 0.0;
        *t2 = 0.0;
        saxpy<<<num_blocks, block_size_1d>>>(x, x, p, alpha, N, 0);
        cudaDeviceSynchronize();
        saxpy<<<num_blocks, block_size_1d>>>(r, r, y, -1.0 * alpha, N, 0);
        cudaDeviceSynchronize();
        l2_norm<<<num_blocks, block_size_1d>>>(r, t1, N, 0);
        cudaDeviceSynchronize();
        float beta = *t1 / old_t1;
        saxpy<<<num_blocks, block_size_1d>>>(p, r, p, beta, N, 0);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    // Other implementation;
    // for (int i = 0; i < P; i++) {
    //     matrix_vector_mult_axpy<<<num_blocks, block_size_1d>>>(A[i], x, b, -1, r, std::min(S, N - i * S), N, i * S);
    //     cudaDeviceSynchronize();
    //     cpy<<<num_blocks, block_size_1d>>>(p, r, std::min(S, N - i * S), i * S);
    //     cudaDeviceSynchronize();
    //     l2_norm<<<num_blocks, block_size_1d>>>(r, t1[i], std::min(S, N - i * S), i * S);
    //     cudaDeviceSynchronize();
    //     t1_tot += t1[i][0];
    // }   
    // for (int iter = 0; iter < ITER; iter++) {
    //     for (int i = 0; i < P; i++) {
    //         matrix_vector_mult<<<num_blocks, block_size_1d>>>(A[i], p, y, std::min(S, N - i * S), N, i * S);
    //         cudaDeviceSynchronize();
    //         dot<<<num_blocks, block_size_1d>>>(p, y, t2[i], std::min(S, N - i * S), i * S);
    //         cudaDeviceSynchronize();
    //         t2_tot += t2[i][0];
    //     }
    //     float alpha = t1_tot / t2_tot;
    //     float old_t1 = t1_tot;
    //     t1_tot = 0.0;
    //     t2_tot = 0.0;
    //     for (int i = 0; i < P; i++) {
    //         saxpy<<<num_blocks, block_size_1d>>>(x, x, p, alpha, std::min(S, N - i * S), i * S);
    //         cudaDeviceSynchronize();
    //         saxpy<<<num_blocks, block_size_1d>>>(r, r, y, -1.0 * alpha, std::min(S, N - i * S), i * S);
    //         cudaDeviceSynchronize();
    //         t1[i][0] = 0;
    //         l2_norm<<<num_blocks, block_size_1d>>>(r, t1[i], std::min(S, N - i * S), i * S);
    //         cudaDeviceSynchronize();
    //         t1_tot += t1[i][0];
    //     }
    //     float beta = t1_tot / old_t1;
    //     for (int i = 0; i < P; i++) {
    //         saxpy<<<num_blocks, block_size_1d>>>(p, r, p, beta, std::min(S, N - i * S), i * S);
    //         cudaDeviceSynchronize();
    //         t2[i][0] = 0;
    //     }
    // }
    // cudaDeviceSynchronize();
}

void Benchmark9M::execute_async(int iter) {
    if (pascalGpu && do_prefetch) {
        for (int i = 0; i < P; i++) {
            cudaSetDevice(select_gpu(i, max_devices));
            cudaMemPrefetchAsync(A[i], sizeof(float) * S * N, 0, s[i]);
        }
        cudaSetDevice(select_gpu(0, max_devices));
        cudaMemPrefetchAsync(x, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(b, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(r, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(p, sizeof(float) * N, 0, s1);
    }

    cudaEvent_t e[P];
    for (int i = 0; i < P; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        precondition<<<num_blocks, block_size_1d, 0, s[i]>>>(A[i], N, std::min(S, N - i * S), i * S);
        matrix_vector_mult_axpy<<<num_blocks, block_size_1d, 0, s[i]>>>(A[i], x, b, -1, r, std::min(S, N - i * S), N, i * S);
        cudaEventCreate(&e[i]);
        cudaEventRecord(e[i], s[i]);
    }
    cudaSetDevice(select_gpu(0, max_devices));
    for (int i = 0; i < P; i++) {
        cudaStreamWaitEvent(s1, e[i], 0);
    }
    cudaEvent_t e_c;
    cudaEventCreate(&e_c);
    cpy<<<num_blocks, block_size_1d, 0, s1>>>(p, r, N, 0);
    cudaEventRecord(e_c, s1);
    for (int i = 0; i < P; i++) {
        cudaStreamWaitEvent(s2, e[i], 0);
    }
    l2_norm<<<num_blocks, block_size_1d, 0, s2>>>(r, t1, N, 0);
    for (int iter = 0; iter < ITER; iter++) {
        cudaEvent_t e2[P];
        for (int i = 0; i < P; i++) {
            cudaSetDevice(select_gpu(i, max_devices));
            cudaStreamWaitEvent(s[i], e_c, 0);
            matrix_vector_mult<<<num_blocks, block_size_1d, 0, s[i]>>>(A[i], p, y, std::min(S, N - i * S), N, i * S);
            cudaEventCreate(&e2[i]);
            cudaEventRecord(e2[i], s[i]);
        }
        cudaSetDevice(select_gpu(0, max_devices));
        for (int i = 0; i < P; i++) {
            cudaStreamWaitEvent(s1, e2[i], 0);
        }
        dot<<<num_blocks, block_size_1d, 0, s1>>>(p, y, t2, N, 0);
        cudaStreamSynchronize(s1);
        cudaStreamSynchronize(s2);
        float alpha = *t1 / *t2;
        float old_t1 = *t1;
        *t1 = 0.0;
        *t2 = 0.0;
        saxpy<<<num_blocks, block_size_1d, 0, s1>>>(x, x, p, alpha, N, 0);
        saxpy<<<num_blocks, block_size_1d, 0, s2>>>(r, r, y, -1.0 * alpha, N, 0);
        l2_norm<<<num_blocks, block_size_1d, 0, s2>>>(r, t1, N, 0);
        cudaStreamSynchronize(s2);
        float beta = *t1 / old_t1;
        saxpy<<<num_blocks, block_size_1d, 0, s1>>>(p, r, p, beta, N, 0);
    }
    cudaStreamSynchronize(s1);

    // Other implementation;
    // for (int i = 0; i < P; i++) {
    //     cudaSetDevice(select_gpu(i, max_devices));
    //     matrix_vector_mult_axpy<<<num_blocks, block_size_1d, 0, s[i]>>>(A[i], x, b, -1, r, std::min(S, N - i * S), N, i * S);
    //     cpy<<<num_blocks, block_size_1d, 0, s[i]>>>(p, r, std::min(S, N - i * S), i * S);
    //     l2_norm<<<num_blocks, block_size_1d, 0, s[i]>>>(r, t1[i], std::min(S, N - i * S), i * S);
    // }
    // for (int i = 0; i < P; i++) {
    //     cudaStreamSynchronize(s[i]);
    //     t1_tot += t1[i][0];
    // }

    // for (int iter = 0; iter < ITER; iter++) {
    //     for (int i = 0; i < P; i++) {
    //         cudaSetDevice(select_gpu(i, max_devices));
    //         matrix_vector_mult<<<num_blocks, block_size_1d, 0, s[i]>>>(A[i], p, y, std::min(S, N - i * S), N, i * S);
    //         dot<<<num_blocks, block_size_1d, 0, s[i]>>>(p, y, t2[i], std::min(S, N - i * S), i * S);
    //     }
    //     for (int i = 0; i < P; i++) {
    //         cudaStreamSynchronize(s[i]);
    //         t2_tot += t2[i][0];
    //     }
    //     float alpha = t1_tot / t2_tot;
    //     float old_t1 = t1_tot;
    //     t1_tot = 0.0;
    //     t2_tot = 0.0;
    //     for (int i = 0; i < P; i++) {
    //         cudaSetDevice(select_gpu(i, max_devices));
    //         t1[i][0] = 0;
    //         saxpy<<<num_blocks, block_size_1d, 0, s[i]>>>(x, x, p, alpha, std::min(S, N - i * S), i * S);
    //         saxpy<<<num_blocks, block_size_1d, 0, s[i]>>>(r, r, y, -1.0 * alpha, std::min(S, N - i * S), i * S);
    //         l2_norm<<<num_blocks, block_size_1d, 0, s[i]>>>(r, t1[i], std::min(S, N - i * S), i * S);
    //     }
    //     for (int i = 0; i < P; i++) {
    //         cudaStreamSynchronize(s[i]);
    //         t1_tot += t1[i][0];
    //     }
    //     float beta = t1_tot / old_t1;
    //     for (int i = 0; i < P; i++) {
    //         cudaSetDevice(select_gpu(i, max_devices));
    //         saxpy<<<num_blocks, block_size_1d, 0, s[i]>>>(p, r, p, beta, std::min(S, N - i * S), i * S);
    //     }
    //     for (int i = 0; i < P; i++) {
    //         cudaStreamSynchronize(s[i]);
    //         t2[i][0] = 0;
    //     }
    // }
}

std::string Benchmark9M::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(x[0]);
    } else {
        for (int i = 0; i < P; i++) {
            matrix_vector_mult_axpy<<<num_blocks, block_size_1d>>>(A[i], x, b, -1, y, std::min(S, N - i * S), N, i * S);
        }
        cudaDeviceSynchronize();
        std::string res = "[";
        for (int j = 0; j < std::min(10, N); j++) {
            res += std::to_string(y[j]) + ", ";
        }

        float sum = 0;
        for (int j = 0; j < N; j++) {
            sum += y[j];
        }
        return res + "...], sum=" + std::to_string(sum);
    }
}