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

#include "b6.cuh"

#define BLOCK_SIZE_V100 64 // Just a recommendation of optimal block size for the V100;
#define P 16

//////////////////////////////
//////////////////////////////

extern "C" __global__ void nb_1_m(const int* x, const float* y, float* z, int n, int partition_rows, int n_feat, int n_classes, int partition_num) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < min(partition_rows, n - partition_num * partition_rows); i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_classes; j++) {
            for (int q = 0; q < n_feat; q++) {
                z[partition_num * partition_rows * n_classes + i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
            }
        }
    }
}

extern "C" __global__ void nb_2_m(const float* x, float* y, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float curr_max = x[i * n_col_x];
        for (int j = 0; j < n_col_x; j++) {
            curr_max = fmaxf(curr_max, x[i * n_col_x + j]);
        }
        y[i] = curr_max;
    }
}

extern "C" __global__ void nb_3_m(const float* x, const float* y, float* z, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < n_col_x; j++) {
            sum += expf(x[i * n_col_x + j] - y[i]);
        }
        z[i] = logf(sum) + y[i];
    }
}

extern "C" __global__ void nb_4_m(float* x, const float* y, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] = expf(x[i * n_col_x + j] - y[i]);
        }
    }
}

// extern "C" __global__ void rr_1_m(const int* x, float* sum, float *sum_squared, int n_row_x, int n_col_x) {
//     for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {
//         float feature_mean = 0;
//         float sum_sq = 0;
//         // Compute mean and variance;
//         for (int i = 0; i < n_row_x; i++) {
//             float x_tmp = x[j * n_row_x + i];
//             feature_mean += x_tmp;
//             sum_sq += x_tmp * x_tmp;
//         }
//         sum[j] += feature_mean;
//         sum_squared[j] += sum_sq;
//     }
// }

// extern "C" __global__ void rr_1_2_m(const int *x, float *y, const float* sum, const float *sum_squared, int n_row_x, int n_col_x, int partition, int partition_size) {
//     // Normalize each row;
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < partition_size; i += blockDim.x * gridDim.x) {
//         for (int j = 0; j < n_col_x; j++) {
//             float mean = sum[j] / n_row_x;
//             float std = sqrtf(sum_squared[j] / n_row_x - mean * mean);
//             y[partition * partition_size * n_col_x + i * n_col_x + j] = (x[i * n_col_x + j] - mean) / std;
//         }
//     }
// }

extern "C" __global__ void rr_1_m(const int* x, float* mean, float *std, int n_row_x, int n_col_x, int partition, int partition_size) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {
        float feature_mean = 0;
        float sum_sq = 0;
        // Compute mean and variance;
        for (int i = 0; i < partition_size; i++) {
            float x_tmp = x[j * partition_size + i];
            feature_mean += x_tmp;
            sum_sq += x_tmp * x_tmp;
        }
        // feature_mean /= n_row_x;
        // std[j] = sqrtf(sum_sq / n_row_x - feature_mean * feature_mean);
        // mean[j] = feature_mean;

        // Keep just the sum and squared sum, compute mean and std later;
        mean[j] += feature_mean;
        std[j] += sum_sq;
    }
}

// extern "C" __global__ void rr_1_m(const int* x, float* mean, float *std, int n_row_x, int n_col_x, int partition, int partition_size) {
//     extern __shared__ volatile float scratch[];
//     if (threadIdx.x == 0) {
//         for (int k = 0; k < blockDim.x; k++) { 
//             scratch[k] = 0;
//         }
//     }
//     __syncthreads();
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < partition_size; i += blockDim.x * gridDim.x) {
//         // Compute sum and sum of squares for mean and variance;
//         for (int j = 0; j < n_col_x; j++) {
//             float x_tmp = x[i * n_col_x + j];
//             scratch[threadIdx.x] = x_tmp;
//             // We read blockDim.x values at the same time, let the first thread do the reduction;
//             __syncthreads();
//             if (threadIdx.x == 0) {
//                 float mean_tmp = 0;
//                 float std_tmp = 0;
//                 for (int k = 0; k < blockDim.x; k++) { 
//                     mean_tmp += scratch[k];
//                     std_tmp += scratch[k] * scratch[k];
//                 }
//                 mean[j] += mean_tmp;
//                 std[j] += std_tmp;
//             }
//         }
//     }
// }

extern "C" __global__ void rr_1_1_m(float* mean, float *std, const float *mean_curr, const float *std_curr, int n_row_x, int n_col_x, int partition_index, int partition_size) {
    // We use partition 0 to accumulate, so skip it;
    if (partition_index == 0) return;

    // Aggregate mean and std from different partitions;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_col_x; i += blockDim.x * gridDim.x) {
        mean[i] += mean_curr[i];
        std[i] += std_curr[i];
        // When processing the last partition, compute the final mean and std;
        if (partition_index == P - 1) {
            mean[i] /= n_row_x;
            std[i] = sqrtf(std[i] / n_row_x - mean[i] * mean[i]);
        }
    }
}

extern "C" __global__ void rr_1_2_m(const int *x, float *y, const float* mean, const float *std, int n_row_x, int n_col_x, int partition_size) {
    // Normalize each row;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < partition_size; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            // if (i == 0) printf("[i=%d][j=%d] mean=%f std=%f\n", i, j, mean[j], mean[j] * mean[j]);
            // float mean_curr = mean[j] / n_row_x;
            // float std_curr = sqrtf(std[j] / n_row_x - mean_curr * mean_curr);
            float mean_curr = mean[j];
            float std_curr = std[j];
            y[i * n_col_x + j] = (x[i * n_col_x + j] - mean_curr) / std_curr;
        }
    }
}

extern "C" __global__ void rr_2_m(const float* x, const float* y, float* z, int n, int partition_rows, int n_feat, int n_classes, int partition_num) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < min(partition_rows, n - partition_num * partition_rows); i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_classes; j++) {
            for (int q = 0; q < n_feat; q++) {
                z[partition_num * partition_rows * n_classes + i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
            }
        }
    }
}

extern "C" __global__ void rr_3_m(float* x, const float* y, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] += y[j];
        }
    }
}

extern "C" __global__ void softmax_m(float* x, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float row_exp_sum = 0;
        for (int j = 0; j < n_col_x; j++) {
            row_exp_sum += expf(x[i * n_col_x + j]);
        }
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] = expf(x[i * n_col_x + j]) / row_exp_sum;
        }
    }
}

extern "C" __global__ void argmax_m(const float* x, const float* y, int* z, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        int curr_best_index = 0;
        float curr_best = x[i * n_col_x] + y[i * n_col_x];
        for (int j = 0; j < n_col_x; j++) {
            float curr = x[i * n_col_x + j] + y[i * n_col_x + j];
            if (curr > curr_best) {
                curr_best = curr;
                curr_best_index = j;
            }
        }
        z[i] = curr_best_index;
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark6M::alloc() {
    S = (N + P - 1) / P;
    x = (int**) malloc(sizeof(int*) * P);
    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&x[i], sizeof(int) * S * num_features);
    }
    err = cudaMallocManaged(&x_o, sizeof(int) * N * num_features);
    z = (float**) malloc(sizeof(float*) * P);
    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&z[i], sizeof(float) * S * num_features);
    }
    mean = (float**) malloc(sizeof(float*) * P);
    std = (float**) malloc(sizeof(float*) * P);
    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&mean[i], sizeof(float) * num_features);
        err = cudaMallocManaged(&std[i], sizeof(float) * num_features);
    }
    err = cudaMallocManaged(&nb_feat_log_prob, sizeof(float) * num_classes * num_features);
    err = cudaMallocManaged(&nb_class_log_prior, sizeof(float) * num_classes);
    err = cudaMallocManaged(&ridge_coeff, sizeof(float) * num_classes * num_features);
    err = cudaMallocManaged(&ridge_intercept, sizeof(float) * num_classes);
    err = cudaMallocManaged(&nb_amax, sizeof(float) * N);
    err = cudaMallocManaged(&nb_l, sizeof(float) * N);
    err = cudaMallocManaged(&r1, sizeof(float) * N * num_classes);
    err = cudaMallocManaged(&r2, sizeof(float) * N * num_classes);
    err = cudaMallocManaged(&r, sizeof(int) * N);

    // Stream 0;
    int gpu = select_gpu(0, max_devices);
    cudaSetDevice(gpu);
    err = cudaStreamCreate(&s1);
    // Stream 1;
    gpu = select_gpu(1, max_devices);
    cudaSetDevice(gpu);
    err = cudaStreamCreate(&s2);
    // Other streams;
    s_n = (cudaStream_t *) malloc(sizeof(cudaStream_t) * P);
    s_r = (cudaStream_t *) malloc(sizeof(cudaStream_t) * P);
    for (int i = 0; i < P; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        err = cudaStreamCreate(&s_n[i]);
        err = cudaStreamCreate(&s_r[i]);
    }
}

void Benchmark6M::init() {
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < num_features; j++) {
            nb_feat_log_prob[i * num_features + j] = (float)(rand()) / (float)(RAND_MAX);
            ridge_coeff[i * num_features + j] = (float)(rand()) / (float)(RAND_MAX);
        }
        nb_class_log_prior[i] = (float)(rand()) / (float)(RAND_MAX);
        ridge_intercept[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < num_features; j++) {
            x_o[i * num_features + j] = rand() % max_occurrence_of_ngram;
        }
        for (int j = 0; j < num_classes; j++) {
            r1[i * num_classes + j] = nb_class_log_prior[j];
            r2[i * num_classes + j] = 0;
        }
    }
    for (int p = 0; p < P; p++) {
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < num_features; j++) {
                int index = p * S * num_features + i * num_features + j;
                if (index < N * num_features) {
                    x[p][i * num_features + j] = x_o[index];
                } else {
                    x[p][i * num_features + j] = 0;
                }
            }
        }
    }
}

void Benchmark6M::reset() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < num_classes; j++) {
            r1[i * num_classes + j] = nb_class_log_prior[j];
            r2[i * num_classes + j] = 0;
        }
    }
    for (int p = 0; p < P; p++) {
        for (int i = 0; i < num_features; i++) {
            mean[p][i] = 0;
            std[p][i] = 0;
        }
    }
}

void Benchmark6M::execute_sync(int iter) {
    if (do_prefetch && pascalGpu) {
        cudaMemPrefetchAsync(r1, sizeof(float) * N * num_classes, 0, 0);
        cudaMemPrefetchAsync(r2, sizeof(float) * N * num_classes, 0, 0);
        cudaMemPrefetchAsync(r, sizeof(int) * N, 0, 0);
    }
    for (int i = 0; i < P; i++) {
        rr_1_m<<<num_blocks, block_size_1d>>>(x[i], mean[i], std[i], N, num_features, i, S);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < P; i++) {
        rr_1_1_m<<<num_blocks, block_size_1d>>>(mean[0], std[0], mean[i], std[i], N, num_features, i, S);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < P; i++) {
        rr_1_2_m<<<num_blocks, block_size_1d>>>(x[i], z[i], mean[0], std[0], N, num_features, S);
        cudaDeviceSynchronize();
        rr_2_m<<<num_blocks, block_size_1d>>>(z[i], ridge_coeff, r2, N, S, num_features, num_classes, i);
        cudaDeviceSynchronize();
    }
    rr_3_m<<<num_blocks, block_size_1d>>>(r2, ridge_intercept, N, num_classes);
    cudaDeviceSynchronize();
    softmax_m<<<num_blocks, block_size_1d>>>(r2, N, num_classes);
    cudaDeviceSynchronize();

    for (int i = 0; i < P; i++) {
        nb_1_m<<<num_blocks, block_size_1d>>>(x[i], nb_feat_log_prob, r1, N, S, num_features, num_classes, i);
        cudaDeviceSynchronize();
    }
    nb_2_m<<<num_blocks, block_size_1d>>>(r1, nb_amax, N, num_classes);
    cudaDeviceSynchronize();
    nb_3_m<<<num_blocks, block_size_1d>>>(r1, nb_amax, nb_l, N, num_classes);
    cudaDeviceSynchronize();
    nb_4_m<<<num_blocks, block_size_1d>>>(r1, nb_l, N, num_classes);
    cudaDeviceSynchronize();
    softmax_m<<<num_blocks, block_size_1d>>>(r1, N, num_classes);
    cudaDeviceSynchronize();

    argmax_m<<<num_blocks, block_size_1d>>>(r1, r2, r, N, num_classes);
    cudaDeviceSynchronize();
}

void Benchmark6M::execute_async(int iter) {

    // RR;
    int gpu = select_gpu(0, max_devices);
    cudaSetDevice(gpu);
    if (!pascalGpu || stream_attach) {
        cudaStreamAttachMemAsync(s1, ridge_coeff, 0);
        cudaStreamAttachMemAsync(s1, r2, 0);
        cudaStreamAttachMemAsync(s1, ridge_intercept, 0);
    }
    if (do_prefetch && pascalGpu) {
        cudaMemPrefetchAsync(r2, sizeof(float) * N * num_classes, gpu, s1);
        cudaMemPrefetchAsync(r, sizeof(int) * N, gpu, s1);
    }
    cudaEvent_t e_r0[P];
    for (int i = 0; i < P; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        rr_1_m<<<num_blocks, block_size_1d, 0, s_r[i]>>>(x[i], mean[i], std[i], N, num_features, i, S);
        cudaEventCreate(&e_r0[i]);
        cudaEventRecord(e_r0[i], s_r[i]);
    }
    cudaSetDevice(select_gpu(0, max_devices));
    for (int i = 0; i < P; i++) {
        cudaStreamWaitEvent(s1, e_r0[i], 0);
        rr_1_1_m<<<num_blocks, block_size_1d, 0, s1>>>(mean[0], std[0], mean[i], std[i], N, num_features, i, S);
    }
    cudaEvent_t e1;
    cudaEventCreate(&e1);
    cudaEventRecord(e1, s1);
    cudaEvent_t e_r1[P];
    for (int i = 0; i < P; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        cudaStreamWaitEvent(s_r[i], e1, 0);
        rr_1_2_m<<<num_blocks, block_size_1d, 0, s_r[i]>>>(x[i], z[i], mean[0], std[0], N, num_features, S);
        rr_2_m<<<num_blocks, block_size_1d, 0, s_r[i]>>>(z[i], ridge_coeff, r2, N, S, num_features, num_classes, i);
        cudaEventCreate(&e_r1[i]);
        cudaEventRecord(e_r1[i], s_r[i]);
    }
    // Stream 1 waits all other streams;
    cudaSetDevice(gpu);
    for (int i = 0; i < P; i++) {
        cudaStreamWaitEvent(s1, e_r1[i], 0);
    }
    rr_3_m<<<num_blocks, block_size_1d, 0, s1>>>(r2, ridge_intercept, N, num_classes);
    softmax_m<<<num_blocks, block_size_1d, 0, s1>>>(r2, N, num_classes);

    // NB;
    gpu = select_gpu(1, max_devices);
    cudaSetDevice(gpu);
    if (!pascalGpu || stream_attach) {
        cudaStreamAttachMemAsync(s2, nb_feat_log_prob, 0);
        cudaStreamAttachMemAsync(s2, r1, 0);
        cudaStreamAttachMemAsync(s2, nb_amax, 0);
        cudaStreamAttachMemAsync(s2, nb_l, 0);
    }
    if (do_prefetch && pascalGpu) {
        cudaMemPrefetchAsync(r1, sizeof(float) * N * num_classes, gpu, s2);
    }
    cudaEvent_t e2;
    cudaEventCreate(&e2);
    cudaEventRecord(e2, s2);
    cudaEvent_t e_n[P];
    for (int i = 0; i < P; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        cudaStreamWaitEvent(s_n[i], e2, 0);
        nb_1_m<<<num_blocks, block_size_1d, 0, s_n[i]>>>(x[i], nb_feat_log_prob, r1, N, S, num_features, num_classes, i);
        cudaEventCreate(&e_n[i]);
        cudaEventRecord(e_n[i], s_n[i]);
    }
    // Stream 2 waits all other streams;
    cudaSetDevice(gpu);
    for (int i = 0; i < P; i++) {
        cudaStreamWaitEvent(s2, e_n[i], 0);
    }
    nb_2_m<<<num_blocks, block_size_1d, 0, s2>>>(r1, nb_amax, N, num_classes);
    nb_3_m<<<num_blocks, block_size_1d, 0, s2>>>(r1, nb_amax, nb_l, N, num_classes);
    nb_4_m<<<num_blocks, block_size_1d, 0, s2>>>(r1, nb_l, N, num_classes);
    softmax_m<<<num_blocks, block_size_1d, 0, s2>>>(r1, N, num_classes);

    // Stream 1 waits stream 2;
    cudaEvent_t e3;
    cudaEventCreate(&e3);
    cudaEventRecord(e3, s2);
    cudaSetDevice(select_gpu(0, max_devices));
    cudaStreamWaitEvent(s1, e3, 0);
    argmax_m<<<num_blocks, block_size_1d, 0, s1>>>(r1, r2, r, N, num_classes);
    cudaDeviceSynchronize();
}

std::string Benchmark6M::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(r[0]);
    } else {
        std::string res = "[";
        for (int j = 0; j < 10; j++) {
            res += std::to_string(r[j]) + ", ";
        }

        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += r[j];
        }
        return res + "...], sum=" + std::to_string(sum);
    }
}