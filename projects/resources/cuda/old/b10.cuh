#pragma once


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
                    kernel_local[l + L * (j + K * (i + K * m))] = kernels[l + L * (j  + K * (i + K * m))];
                }
            }
        }
    }
    __syncthreads();
    
   
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (int) ceilf((float) N / stride) - radius; i += blockDim.x * gridDim.x) {
        int out_index = M * i / stride;
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < (int) ceilf((float) M / stride) - radius; j += blockDim.y * gridDim.y) {
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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (int) ceilf((float) N / stride) - radius; i += blockDim.x * gridDim.x) {
        int out_index = M * i / stride;
        int i_f = i * stride + radius;
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < (int) ceilf((float) M / stride) - radius; j += blockDim.y * gridDim.y) {
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
    for(int i = threadIdx.x; i < L; i += blockDim.x) {
        out_local[i] = 0;
    }
    __syncthreads();
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < M; j += blockDim.y * gridDim.y) {
            for (int l = 0; l < L; l++) {   
                atomicAdd(out_local + l, x[l + L * (j + M * i)]);
            }
        }
    }
    __syncthreads();
    for(int l = threadIdx.x; l < L; l += blockDim.x) {
        atomicAdd(out + l, out_local[l] / (M * N));
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void dot_product(const float *x, const float *y, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] * y[i];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

extern "C" __global__ void concat(float *z, const float *x, const float *y, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        z[i] = x[i];
        z[i + n] = y[i];
    }
}

inline void reset(float *x, float *y, float *x_cpu, float *y_cpu, int N, float *res) {
    for (int i = 0; i < N; i++) {
        x[i] = x_cpu[i];
        y[i] = y_cpu[i];
    }
    *res = 0;
}

