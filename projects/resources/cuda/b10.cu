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

#include "b10.cuh"

//////////////////////////////
//////////////////////////////

#define NUM_THREADS_PER_BLOCK_2D 8
#define NUM_THREADS_PER_BLOCK 32
#define WARP_SIZE 32
#define NUM_BLOCKS 16

extern "C" __global__ void conv2d(float *out, float *x, float *kernels, int N, int M, int L, int K, int k_out, int stride) {
    extern __shared__ float kernel_local[];
    int radius = K / 2;

    for (int m = 0; m < k_out; m++) {
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            for (int j = threadIdx.y; j < K; j += blockDim.y) {
                for (int l = 0; l < L; l++) {
                    kernel_local[l + L * (j + K * (i + K * m))] = kernels[l + L * (j + K * (i + K * m))];
                }
            }
        }
    }
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (int)ceilf((float)N / stride) - radius; i += blockDim.x * gridDim.x) {
        int out_index = M * i / stride;
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < (int)ceilf((float)M / stride) - radius; j += blockDim.y * gridDim.y) {
            for (int m = 0; m < k_out; m++) {
                // for (int m = blockIdx.z * blockDim.z + threadIdx.z; m < k_out; m += blockDim.z * gridDim.z) {
                float res = 0;
                int i_f = i * stride + radius;
                int j_f = j * stride + radius;
                for (int k_i = -radius; k_i <= radius; k_i++) {
                    for (int k_j = -radius; k_j <= radius; k_j++) {
                        int kernel_index = (k_j + radius + K * (k_i + radius + K * m));
                        for (int l = 0; l < L; l++) {
                            int ni = i_f + k_i;
                            int nj = j_f + k_j;
                            res += kernel_local[l + L * kernel_index] * x[((ni * M) + nj) * L + l];
                        }
                    }
                }
                // Apply ReLU operator;
                out[m + k_out * (j + out_index)] = max(res, 0.0);
            }
        }
    }
}

extern "C" __global__ void mean_pooling(float *out, float *x, int N, int M, int L, int K, int stride) {
    int radius = K / 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (int)ceilf((float)N / stride) - radius; i += blockDim.x * gridDim.x) {
        int out_index = M * i / stride;
        int i_f = i * stride + radius;
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < (int)ceilf((float)M / stride) - radius; j += blockDim.y * gridDim.y) {
            int j_f = j * stride + radius;
            for (int l = blockIdx.z * blockDim.z + threadIdx.z; l < L; l += blockDim.z * gridDim.z) {
                float res = 0;
                for (int k_i = -radius; k_i <= radius; k_i++) {
                    int ni = i_f + k_i;
                    for (int k_j = -radius; k_j <= radius; k_j++) {
                        int nj = j_f + k_j;
                        res += x[((ni * M) + nj) * L + l];
                    }
                }
                // Apply mean operator;
                out[l + L * (j + out_index)] = res / (K * K);
            }
        }
    }
}

extern "C" __global__ void gap(float *out, float *x, int N, int M, int L) {
    extern __shared__ float out_local[];
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        out_local[i] = 0;
    }
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < M; j += blockDim.y * gridDim.y) {
            for (int l = 0; l < L; l++) {
                atomicAdd(out_local + l, x[l + L * (j + M * i)]);
            }
        }
    }
    __syncthreads();
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        atomicAdd(out + l, out_local[l] / (M * N));
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void dot_product(const float *x, const float *y, float *z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] * y[i];
    }
    sum = warp_reduce(sum);                    // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum);                     // The first thread in the warp updates the output;
}

extern "C" __global__ void concat(float *z, const float *x, const float *y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        z[i] = x[i];
        z[i + n] = y[i];
    }
}

// inline void reset(float *x, float *y, float *x_cpu, float *y_cpu, int N, float *res) {
//     for (int i = 0; i < N; i++) {
//         x[i] = x_cpu[i];
//         y[i] = y_cpu[i];
//     }
//     *res = 0;
// }

//////////////////////////////
//////////////////////////////

void Benchmark10::alloc() {
    x_cpu = (float *)malloc(sizeof(float) * N * N * channels);
    y_cpu = (float *)malloc(sizeof(float) * N * N * channels);
    x_len = N * N * channels;
    x1_len = (N / stride) * (N / stride) * kn1;
    pooled_len = x1_len / (pooling_diameter * pooling_diameter);
    x2_len = ((N / stride) / pooling_diameter / stride) * ((N / stride) / pooling_diameter / stride) * kn2;
    x3_len = kn2;

    err = cudaMallocManaged(&x, sizeof(float) * x_len);
    err = cudaMallocManaged(&x1, sizeof(float) * x1_len);
    err = cudaMallocManaged(&x2, sizeof(float) * x2_len);
    err = cudaMallocManaged(&x3, sizeof(float) * x3_len);

    err = cudaMallocManaged(&y, sizeof(float) * x_len);
    err = cudaMallocManaged(&y1, sizeof(float) * x1_len);
    err = cudaMallocManaged(&y2, sizeof(float) * x2_len);
    err = cudaMallocManaged(&y3, sizeof(float) * x3_len);

    k1_len = channels * K * K * kn1;
    k2_len = kn1 * K * K * kn2;
    err = cudaMallocManaged(&kernel_1, sizeof(float) * k1_len);
    err = cudaMallocManaged(&kernel_2, sizeof(float) * k2_len);
    err = cudaMallocManaged(&kernel_3, sizeof(float) * k1_len);
    err = cudaMallocManaged(&kernel_4, sizeof(float) * k2_len);

    z_len = 2 * x2_len;
    err = cudaMallocManaged(&z, sizeof(float) * z_len);
    err = cudaMallocManaged(&dense_weights, sizeof(float) * z_len);
    err = cudaMallocManaged(&res, sizeof(float));

    err = cudaMallocManaged(&x11, sizeof(float) * pooled_len);
    err = cudaMallocManaged(&y11, sizeof(float) * pooled_len);

    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);
}

void Benchmark10::init() {
    for (int i = 0; i < x_len; i++) {
        x_cpu[i] = (float)(rand()) / (float)(RAND_MAX);
        y_cpu[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    for (int i = 0; i < k1_len; i++) {
        kernel_1[i] = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
        kernel_3[i] = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
    }
    for (int i = 0; i < k2_len; i++) {
        kernel_2[i] = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
        kernel_4[i] = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
    }

    for (int i = 0; i < z_len; i++) {
        dense_weights[i] = (((float)(rand()) / (float)(RAND_MAX)) * 2 - 1) / z_len;
    }
}

void Benchmark10::reset() {
    for (int i = 0; i < x_len; i++) {
        x[i] = x_cpu[i];
        y[i] = y_cpu[i];
    }
    *res = 0;
}

void Benchmark10::execute_sync(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks, num_blocks);
    dim3 grid_size_2(num_blocks / 2, num_blocks / 2);

    dim3 block_size_3d_dim(block_size_2d / 2, block_size_2d / 2, block_size_2d / 2);
    dim3 grid_size_3(num_blocks / 2, num_blocks / 2, num_blocks / 2);

    if (do_prefetch && pascalGpu) {
        cudaMemPrefetchAsync(x, sizeof(float) * x_len, 0, 0);
        cudaMemPrefetchAsync(y, sizeof(float) * x_len, 0, 0);
    }

    conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float)>>>(x1, x, kernel_1, N, N, channels, K, kn1, stride);
    cudaDeviceSynchronize();
    conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float)>>>(y1, y, kernel_3, N, N, channels, K, kn1, stride);
    cudaDeviceSynchronize();

    mean_pooling<<<grid_size_3, block_size_3d_dim>>>(x11, x1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);
    cudaDeviceSynchronize();
    mean_pooling<<<grid_size_3, block_size_3d_dim>>>(y11, y1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);
    cudaDeviceSynchronize();

    conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float)>>>(x2, x11, kernel_2, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);
    cudaDeviceSynchronize();
    conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float)>>>(y2, y11, kernel_4, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);
    cudaDeviceSynchronize();

    // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float)>>>(x2, x1, kernel_2, N / stride, N / stride, kn1, K, kn2, stride);
    // cudaDeviceSynchronize();
    // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float)>>>(y2, y1, kernel_4, N / stride, N / stride, kn1, K, kn2, stride);
    // cudaDeviceSynchronize();

    // gap<<<grid_size_2, block_size_2d_dim, kn2 * sizeof(float)>>>(x3, x2, N / (stride * stride), N / (stride * stride), kn2);
    // cudaDeviceSynchronize();
    // gap<<<grid_size_2, block_size_2d_dim, kn2 * sizeof(float)>>>(y3, y2, N / (stride * stride), N / (stride * stride), kn2);
    // cudaDeviceSynchronize();

    concat<<<num_blocks, block_size_1d>>>(z, x2, y2, x2_len);
    cudaDeviceSynchronize();

    dot_product<<<num_blocks, block_size_1d>>>(z, dense_weights, res, x2_len);
    cudaDeviceSynchronize();
}

void Benchmark10::execute_async(int iter) {
    if (!pascalGpu || stream_attach) {
        cudaStreamAttachMemAsync(s1, x, sizeof(float) * x_len);
        cudaStreamAttachMemAsync(s1, x1, 0);
        cudaStreamAttachMemAsync(s1, x2, 0);
        // cudaStreamAttachMemAsync(s1, x3, 0);
        cudaStreamAttachMemAsync(s1, kernel_1, 0);
        cudaStreamAttachMemAsync(s1, kernel_2, 0);

        cudaStreamAttachMemAsync(s2, y, sizeof(float) * x_len);
        cudaStreamAttachMemAsync(s2, y1, 0);
        // cudaStreamAttachMemAsync(s2, y2, 0);
        // cudaStreamAttachMemAsync(s2, y3, 0);
        cudaStreamAttachMemAsync(s2, kernel_3, 0);
        cudaStreamAttachMemAsync(s2, kernel_4, 0);
    }
    if (do_prefetch && pascalGpu) {
        cudaMemPrefetchAsync(x, sizeof(float) * x_len, 0, 0);
        cudaMemPrefetchAsync(y, sizeof(float) * x_len, 0, 0);
    }
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks, num_blocks);
    dim3 grid_size_2(num_blocks / 2, num_blocks / 2);

    dim3 block_size_3d_dim(block_size_2d / 2, block_size_2d / 2, block_size_2d / 2);
    dim3 grid_size_3(num_blocks / 2, num_blocks / 2, num_blocks / 2);

    conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float), s1>>>(x1, x, kernel_1, N, N, channels, K, kn1, stride);
    conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float), s2>>>(y1, y, kernel_3, N, N, channels, K, kn1, stride);

    mean_pooling<<<grid_size_3, block_size_3d_dim, 0, s1>>>(x11, x1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);
    mean_pooling<<<grid_size_3, block_size_3d_dim, 0, s2>>>(y11, y1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);

    conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s1>>>(x2, x11, kernel_2, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);
    conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s2>>>(y2, y11, kernel_4, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);

    // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s1>>>(x2, x1, kernel_2, N / stride, N / stride, kn1, K, kn2, stride);
    // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s2>>>(y2, y1, kernel_4, N / stride, N / stride, kn1, K, kn2, stride);

    // gap<<<grid_size_2, block_size_2d_dim, kn2 * sizeof(float), s1>>>(x3, x2, N / (stride * stride), N / (stride * stride), kn2);
    // gap<<<grid_size_2, block_size_2d_dim, kn2 * sizeof(float), s2>>>(y3, y2, N / (stride * stride), N / (stride * stride), kn2);

    cudaEvent_t e1;
    cudaEventCreate(&e1);
    cudaEventRecord(e1, s2);
    cudaStreamWaitEvent(s1, e1, 0);

    concat<<<num_blocks, block_size_1d, 0, s1>>>(z, x2, y2, x2_len);

    dot_product<<<num_blocks, block_size_1d, 0, s1>>>(z, dense_weights, res, x2_len);
    cudaStreamSynchronize(s1);
}

void Benchmark10::execute_cudagraph(int iter) {
    if (iter == 0) {
        cudaEvent_t ef;
        cudaEventCreate(&ef);
        cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
        cudaEventRecord(ef, s1);
        cudaStreamWaitEvent(s2, ef, 0);

        dim3 block_size_2d_dim(block_size_2d, block_size_2d);
        dim3 grid_size(num_blocks, num_blocks);
        dim3 grid_size_2(num_blocks / 2, num_blocks / 2);

        dim3 block_size_3d_dim(block_size_2d / 2, block_size_2d / 2, block_size_2d / 2);
        dim3 grid_size_3(num_blocks / 2, num_blocks / 2, num_blocks / 2);

        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float), s1>>>(x1, x, kernel_1, N, N, channels, K, kn1, stride);
        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float), s2>>>(y1, y, kernel_3, N, N, channels, K, kn1, stride);

        mean_pooling<<<grid_size_3, block_size_3d_dim, 0, s1>>>(x11, x1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);
        mean_pooling<<<grid_size_3, block_size_3d_dim, 0, s2>>>(y11, y1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);

        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s1>>>(x2, x11, kernel_2, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);
        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s2>>>(y2, y11, kernel_4, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);

        // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s1>>>(x2, x1, kernel_2, N / stride, N / stride, kn1, K, kn2, stride);
        // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s2>>>(y2, y1, kernel_4, N / stride, N / stride, kn1, K, kn2, stride);

        // gap<<<grid_size_2, block_size_2d_dim, kn2 * sizeof(float), s1>>>(x3, x2, N / (stride * stride), N / (stride * stride), kn2);
        // gap<<<grid_size_2, block_size_2d_dim, kn2 * sizeof(float), s2>>>(y3, y2, N / (stride * stride), N / (stride * stride), kn2);

        cudaEvent_t e1;
        cudaEventCreate(&e1);
        cudaEventRecord(e1, s2);
        cudaStreamWaitEvent(s1, e1, 0);

        concat<<<num_blocks, block_size_1d, 0, s1>>>(z, x2, y2, x2_len);

        dot_product<<<num_blocks, block_size_1d, 0, s1>>>(z, dense_weights, res, x2_len);

        cudaStreamEndCapture(s1, &graph);
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }
    cudaGraphLaunch(graphExec, s1);
    err = cudaStreamSynchronize(s1);
}

void Benchmark10::execute_cudagraph_manual(int iter) {
    if (iter == 0) {
        cudaGraphCreate(&graph, 0);
        int a = N / stride;
        int b = N / stride / pooling_diameter;
        void *kernel_1_args[9] = {(void *)&x1, (void *)&x, (void *)&kernel_1, &N, &N, &channels, &K, &kn1, &stride};
        void *kernel_2_args[9] = {(void *)&y1, (void *)&y, (void *)&kernel_3, &N, &N, &channels, &K, &kn1, &stride};
        void *kernel_3_args[7] = {(void *)&x11, (void *)&x1, &a, &a, &kn1, &pooling_diameter, &pooling_diameter};
        void *kernel_4_args[7] = {(void *)&y11, (void *)&y1, &a, &a, &kn1, &pooling_diameter, &pooling_diameter};
        void *kernel_5_args[9] = {(void *)&x2, (void *)&x11, (void *)&kernel_2, &b, &b, &kn1, &K, &kn2, &stride};
        void *kernel_6_args[9] = {(void *)&y2, (void *)&y11, (void *)&kernel_4, &b, &b, &kn1, &K, &kn2, &stride};
        void *kernel_7_args[4] = {(void *)&z, (void *)&x2, (void *)&y2, &x2_len};
        void *kernel_8_args[4] = {(void *)&z, (void *)&dense_weights, (void *)&res, &x2_len};

        dim3 block_size_2d_dim(block_size_2d, block_size_2d);
        dim3 grid_size(num_blocks, num_blocks);
        dim3 grid_size_2(num_blocks / 2, num_blocks / 2);
        dim3 block_size_3d_dim(block_size_2d / 2, block_size_2d / 2, block_size_2d / 2);
        dim3 grid_size_3(num_blocks / 2, num_blocks / 2, num_blocks / 2);
        dim3 tb(block_size_1d);
        dim3 bs(num_blocks);

        // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float), s1>>>(x1, x, kernel_1, N, N, channels, K, kn1, stride);
        // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float), s2>>>(y1, y, kernel_3, N, N, channels, K, kn1, stride);
        // mean_pooling<<<grid_size_3, block_size_2d_dim, 0, s1>>>(x11, x1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);
        // mean_pooling<<<grid_size_3, block_size_3d_dim, 0, s2>>>(y11, y1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);
        // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s1>>>(x2, x11, kernel_2, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);
        // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s2>>>(y2, y11, kernel_4, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);
        // concat<<<num_blocks, block_size_1d, 0, s1>>>(z, x2, y2, x2_len);
        // dot_product<<<num_blocks, block_size_1d, 0, s1>>>(z, dense_weights, res, x2_len);

        add_node(kernel_1_args, kernel_1_params, (void *)conv2d, grid_size_2, block_size_2d_dim, graph, &k_1, nodeDependencies, K * K * kn1 * channels * sizeof(float));
        add_node(kernel_2_args, kernel_2_params, (void *)conv2d, grid_size_2, block_size_2d_dim, graph, &k_2, nodeDependencies, K * K * kn1 * channels * sizeof(float));

        nodeDependencies.clear();
        nodeDependencies.push_back(k_1);
        add_node(kernel_3_args, kernel_3_params, (void *)mean_pooling, grid_size_3, block_size_2d_dim, graph, &k_3, nodeDependencies);

        nodeDependencies.clear();
        nodeDependencies.push_back(k_2);
        add_node(kernel_4_args, kernel_4_params, (void *)mean_pooling, grid_size_3, block_size_2d_dim, graph, &k_4, nodeDependencies);

        nodeDependencies.clear();
        nodeDependencies.push_back(k_3);
        add_node(kernel_5_args, kernel_5_params, (void *)conv2d, grid_size_2, block_size_2d_dim, graph, &k_5, nodeDependencies, K * K * kn1 * kn2 * sizeof(float));

        nodeDependencies.clear();
        nodeDependencies.push_back(k_4);
        add_node(kernel_6_args, kernel_6_params, (void *)conv2d, grid_size_2, block_size_2d_dim, graph, &k_6, nodeDependencies, K * K * kn1 * kn2 * sizeof(float));

        nodeDependencies.clear();
        nodeDependencies.push_back(k_5);
        nodeDependencies.push_back(k_6);
        add_node(kernel_7_args, kernel_7_params, (void *)concat, bs, tb, graph, &k_7, nodeDependencies);

        nodeDependencies.clear();
        nodeDependencies.push_back(k_7);
        add_node(kernel_8_args, kernel_8_params, (void *)dot_product, bs, tb, graph, &k_8, nodeDependencies);

        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }
    cudaGraphLaunch(graphExec, s1);
    err = cudaStreamSynchronize(s1);
}

void Benchmark10::execute_cudagraph_single(int iter) {
    if (iter == 0) {
        cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);

        dim3 block_size_2d_dim(block_size_2d, block_size_2d);
        dim3 grid_size(num_blocks, num_blocks);
        dim3 grid_size_2(num_blocks / 2, num_blocks / 2);

        dim3 block_size_3d_dim(block_size_2d / 2, block_size_2d / 2, block_size_2d / 2);
        dim3 grid_size_3(num_blocks / 2, num_blocks / 2, num_blocks / 2);

        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float), s1>>>(x1, x, kernel_1, N, N, channels, K, kn1, stride);
        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float), s1>>>(y1, y, kernel_3, N, N, channels, K, kn1, stride);

        mean_pooling<<<grid_size_3, block_size_3d_dim, 0, s1>>>(x11, x1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);
        mean_pooling<<<grid_size_3, block_size_3d_dim, 0, s1>>>(y11, y1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);

        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s1>>>(x2, x11, kernel_2, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);
        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s1>>>(y2, y11, kernel_4, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);

        concat<<<num_blocks, block_size_1d, 0, s1>>>(z, x2, y2, x2_len);

        dot_product<<<num_blocks, block_size_1d, 0, s1>>>(z, dense_weights, res, x2_len);

        cudaStreamEndCapture(s1, &graph);
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }
    cudaGraphLaunch(graphExec, s1);
    err = cudaStreamSynchronize(s1);
}

std::string Benchmark10::print_result(bool short_form) {
    return std::to_string(res[0]);
}
