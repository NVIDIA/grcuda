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

#include "b11.cuh"

//////////////////////////////
//////////////////////////////

#if PARTITION_Z_B11
extern "C" __global__ void matrix_vector_mult_1(const float* x, const float* y, float* z, int n, int m) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[i] = sum;
    }
}

extern "C" __global__ void copy(const float *x, float *y, int n, int offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i + offset] = x[i];
    }
}
#else
extern "C" __global__ void matrix_vector_mult_1(const float* x, const float* y, float* z, int n, int m, int z_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[z_offset + i] = sum;
    }
}
#endif

#define BLOCK_DIM 16
extern "C" __global__ void matrix_vector_mult_2(const float* x, const float* y, float* z, int n, int m, int z_offset) {
    int tile_size = BLOCK_DIM;

    // In the simplest implementation, each block computes a vertical tile of the Z vector, 
    // whose coordinates are given by blockIdx.x;
    // Here, we allow each block to process more tiles, hence the loops below;
    for(int z_tile_i = blockIdx.x; z_tile_i < (m + tile_size - 1) / tile_size; z_tile_i += gridDim.x) {
        // Index of the tile element computed by this thread, with respect of the current tile;
        int z_i = threadIdx.x;
        int z_j = threadIdx.y;
        // Coordinate of the Z matrix element computed by this specific thread, with respect to the overall Z matrix (not counting host-level data partitioning);
        int i = z_tile_i * blockDim.x + threadIdx.x;
        // Value of the Z vector block being computed by this specific thread;
        float z_val_i = 0;
        // Loop over the tiles in the same row of X of the desired output tile in Z;
        for (int curr_tile_index = 0; curr_tile_index < (n + tile_size - 1) / tile_size; curr_tile_index++) {
            // Shared memory used to store the current tiles of X and Y;
            __shared__ float x_tile[BLOCK_DIM][BLOCK_DIM];
            __shared__ float y_tile[BLOCK_DIM];
            // Each thread in the block loads a value into the tile;
            if ((i < n) && (curr_tile_index * tile_size + z_j < m)) {
                x_tile[z_i][z_j] = x[m * i + curr_tile_index * tile_size + z_j];
            } else {
                x_tile[z_i][z_j] = 0;
            }
            if (curr_tile_index * tile_size + z_j < m) {
                y_tile[z_j] = y[curr_tile_index * tile_size + z_j];
            } else {
                y_tile[z_j] = 0;
            }
            // Synchronize threads in the block, ensure the tile has been loaded;
            __syncthreads();
            // Multiply the i row of the tile with the vector tile;
            for (int k = 0; k < tile_size; k++) {   
                z_val_i += x_tile[z_i][k] * y_tile[k];
            }

            // Synchronize threads in the block, ensure the computation has finished before loading the next tile;
            __syncthreads();
        }
        // Write the output value into Z, keeping into account the offset of the current tile;
        if (z_offset + i < n) {
            z[z_offset + i] = z_val_i;
        }     
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark11M::alloc() {
    M = N;
    S = (N + P - 1) / P;
    // x_cpu = (float *) malloc(sizeof(float) * N * M);
    x = (float **) malloc(sizeof(float*) * P);
    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&x[i], sizeof(float) * S * M);
    }
    err = cudaMallocManaged(&y, sizeof(float) * M);
#if PARTITION_Z_B11
    z = (float **) malloc(sizeof(float*) * P);
    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&z[i], sizeof(float) * S);
    }
    cudaMallocManaged(&z_out, sizeof(float) * N);
#else
    err = cudaMallocManaged(&z, sizeof(float) * N);
#endif

    // Create P streams;
    s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * P);
    for (int i = 0; i < P; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        err = cudaStreamCreate(&s[i]);
    }
}

void Benchmark11M::init() {
}

void Benchmark11M::reset() {
    for (int i = 0; i < M; i++) {
        y[i] = float(i + 1) / M; 
    }
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < S * M; j++) {
            x[i][j] = float(i * S * M + j) / (N * M);
        }
    }
}

void Benchmark11M::execute_sync(int iter) {
    if (do_prefetch && pascalGpu) {
        for (int p = 0; p < P; p++) {
            cudaMemPrefetchAsync(x[p], sizeof(float) * S * M, 0, 0);
            cudaDeviceSynchronize();
        }
        cudaMemPrefetchAsync(y, sizeof(float) * M, 0, 0);
    }
    cudaDeviceSynchronize();
    for (int p = 0; p < P; p++) {
#if PARTITION_Z_B11
        matrix_vector_mult_1<<<num_blocks, block_size_1d>>>(x[p], y, z[p], std::min(S, N - p * S), M);
#else
        matrix_vector_mult_1<<<num_blocks, block_size_1d>>>(x[p], y, z, std::min(S, N - p * S), M, p * S);
#endif
        cudaDeviceSynchronize();
    } 
    // Copy data to the output vector;
#if PARTITION_Z_B11
    for (int p = 0; p < P; p++) {
        copy<<<num_blocks, block_size_1d>>>(z[p], z_out, std::min(S, N - p * S), p * S);
    }
#else
    z_out = z;
#endif
    cudaDeviceSynchronize();
}

void Benchmark11M::execute_async(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks, num_blocks);
    for (int p = 0; p < P; p++) {
        cudaSetDevice(select_gpu(p, max_devices));
        if (!pascalGpu || stream_attach) {
            cudaStreamAttachMemAsync(s[p], x[p], sizeof(float) * S * M);
        }
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(x[p], sizeof(float) * S * M, select_gpu(p, max_devices), s[p]);
        }
#if PARTITION_Z_B11
        matrix_vector_mult_1<<<num_blocks, block_size_1d, 0, s[p]>>>(x[p], y, z[p], std::min(S, N - p * S), M);
#else
        matrix_vector_mult_1<<<num_blocks, block_size_1d, 0, s[p]>>>(x[p], y, z, std::min(S, N - p * S), M, p * S);
#endif
        // matrix_vector_mult_2<<<grid_size, block_size_2d_dim, 0, s[p]>>>(x[p], y, z, std::min(S, N - p * S), M, p * S);
    }
    // Copy data to the output vector;
#if PARTITION_Z_B11
    for (int p = 0; p < P; p++) {
        copy<<<num_blocks, block_size_1d, 0, s[p]>>>(z[p], z_out, std::min(S, N - p * S), p * S);
    }
#else
    z_out = z;
#endif

    for (int p = 0; p < P; p++) {
        err = cudaStreamSynchronize(s[p]);
    }
}

std::string Benchmark11M::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(z_out[0]);
    } else {
        std::string res = "[";
        for (int i = 0; i < std::min(100, N); i++) {
            res += std::to_string(z_out[i]) + ", ";
        }
        return res + "...]";
    }
}