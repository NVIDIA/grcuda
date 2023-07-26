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

#include "b13.cuh"

//////////////////////////////
//////////////////////////////

#if PARTITION_Z
// Assume that z is partitioned in blocks;
extern "C" __global__ void matrix_matrix_mult_1(const float* x, const float* y, float* z, int x_num_rows, int x_num_cols, int y_num_cols, int z_num_cols) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < x_num_rows; i += blockDim.x * gridDim.x) {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < y_num_cols; j += blockDim.y * gridDim.y) {
            float sum = 0;
            for (int k = 0; k < x_num_cols; k++) {                
                sum += x[i * x_num_cols + k] * y[k * x_num_cols + j];
            }
            z[i * z_num_cols + j] = sum;
        }
    }
}
#else
// Use a single array for z, but still access it as if it were divided in
extern "C" __global__ void matrix_matrix_mult_1(const float* x, const float* y, float* z, int x_num_rows, int x_num_cols, int y_num_cols, int x_offset, int y_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < x_num_rows; i += blockDim.x * gridDim.x) {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < y_num_cols; j += blockDim.y * gridDim.y) {
            float sum = 0;
            for (int k = 0; k < x_num_cols; k++) {  
                // Y is partioned in vertical bands, and accessed column-major.
                // Here, it is still presented as a horizontal band;             
                sum += x[i * x_num_cols + k] * y[j * x_num_cols  + k];
            }
            z[(x_offset + i) * x_num_cols + (y_offset + j)] = sum;
        }
    }
}

#define BLOCK_DIM 14
// Better implementation, using shared memory to compute square tiles of z;
extern "C" __global__ void matrix_matrix_mult_2(const float* x, const float* y, float* z, int x_num_rows, int x_num_cols, int y_num_cols, int z_num_cols, int x_offset, int y_offset) {

    // int tile_size = BLOCK_DIM;
    int tile_size = blockDim.x;

    // In the simplest implementation, each block computes a tile of the Z matrix, 
    // whose coordinates are given by blockIdx.x and blockIdx.y;
    // Here, we allow each block to process more tiles, hence the loops below;
    for(int z_block_i = blockIdx.x; z_block_i < (x_num_rows + tile_size - 1) / tile_size; z_block_i += gridDim.x) {
        for(int z_block_j = blockIdx.y; z_block_j < (y_num_cols + tile_size - 1) / tile_size; z_block_j += gridDim.y) {
            // Coordinate of the Z matrix element computed by this specific thread, with respect to the current tile;
            int z_i = threadIdx.x;
            int z_j = threadIdx.y;
            // Coordinate of the Z matrix element computed by this specific thread, with respect to the overall Z matrix (not counting host-level data partitioning);
            int i = z_block_i * blockDim.x + threadIdx.x;
            int j = z_block_j * blockDim.y + threadIdx.y;

            // Value of the Z matrix block being computed by this specific thread;
            float z_val_ij = 0;

            // Loop over the tiles in the same row (for X) and column (for Y) of the desired output tile in Z;
            for (int curr_block_index = 0; curr_block_index < (x_num_cols + tile_size - 1) / tile_size; curr_block_index++) {
                // Shared memory used to store the current tiles of X and Y;
                extern __shared__ float tiles[];
                float *x_tile = tiles;
                float *y_tile = tiles + tile_size * tile_size;
                // __shared__ float x_tile[BLOCK_DIM][BLOCK_DIM];
                // __shared__ float y_tile[BLOCK_DIM][BLOCK_DIM];
                // Each thread in the block loads a value into the tile;
                if ((i < x_num_rows) && (curr_block_index * tile_size + z_j < x_num_cols)) {
                    x_tile[z_i * tile_size + z_j] = x[x_num_cols * i + curr_block_index * tile_size + z_j];
                    // x_tile[z_i][z_j] = x[x_num_cols * i + curr_block_index * tile_size + z_j];
                } else {
                    x_tile[z_i * tile_size + z_j] = 0;
                    // x_tile[z_i][z_j] = 0;
                }
                if ((j < y_num_cols) && (curr_block_index * tile_size + z_i < x_num_cols)) {
                    y_tile[z_i * tile_size + z_j] = y[x_num_cols * j + curr_block_index * tile_size + z_i];
                    // y_tile[z_i][z_j] = y[x_num_cols * j + curr_block_index * tile_size + z_i];
                } else {
                    y_tile[z_i * tile_size + z_j] = 0;
                    // y_tile[z_i][z_j] = 0;
                }
                // Synchronize threads in the block, ensure the tile has been loaded;
                __syncthreads();

                // Multiply the i row and j column of the tile;
                for (int k = 0; k < tile_size; k++) {   
                    z_val_ij += x_tile[z_i * tile_size + k] * y_tile[k * tile_size + z_j];
                    // z_val_ij += x_tile[z_i][k] * y_tile[k][z_j];
                }

                // Synchronize threads in the block, ensure the computation has finished before loading the next tile;
                __syncthreads();
            }
            // Write the output value into Z, keeping into account the offset of the current block;
            if (((x_offset + i) < x_num_cols) & (y_offset + j < z_num_cols)) {
                z[(x_offset + i) * x_num_cols + (y_offset + j)] = z_val_ij;
            } 
        }
    }
}
#endif

//////////////////////////////
//////////////////////////////

void Benchmark13M::alloc() {
    S = (N + P - 1) / P;
    PZ = P * P;
    // X is partitioned by rows (horizontal bands), Y is partitioned by columns (vertical bands).
    // Z is partitioned in square blocks.
    // Data are copied into the GPU as row-major for X, and column-major for Y (i.e. Y GPU contains the transpose of Y CPU);
    x_cpu = (float *) malloc(sizeof(float) * N * N);
    y_cpu = (float *) malloc(sizeof(float) * N * N);
    x = (float **) malloc(sizeof(float*) * P);
    y = (float **) malloc(sizeof(float*) * P);
    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&x[i], sizeof(float) * S * N);
        err = cudaMallocManaged(&y[i], sizeof(float) * S * N);
    }
#if PARTITION_Z
    z = (float **) malloc(sizeof(float*) * PZ);
    for (int i = 0; i < PZ; i++) {
        err = cudaMallocManaged(&z[i], sizeof(float) * S * S);
    }
#else
    err = cudaMallocManaged(&z, sizeof(float) * N * N);
#endif
    // Create P * P streams;
#if P2_STREAMS
    s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * P * P);
    for (int p1 = 0; p1 < P; p1++) {
        for (int p2 = 0; p2 < P; p2++) {
            int p = p1 * P + p2;
            cudaSetDevice(select_gpu(p, max_devices));
            err = cudaStreamCreate(&s[p]);
        }
    }    
#else
    s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * P);
    for (int p1 = 0; p1 < P; p1++) {
        cudaSetDevice(select_gpu(p1, max_devices));
        err = cudaStreamCreate(&s[p1]);
    } 
#endif

#if CPU_VALIDATION
    z_cpu = (float*) malloc(sizeof(float) * N * N);
#endif
}

void Benchmark13M::init() {
    // X and Y contains the same data
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x_cpu[i * N + j] = float(i * N + j) / (N * N);
            y_cpu[i * N + j] = float(i * N + j) / (N * N);
        }
    }
}

void Benchmark13M::reset() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int p = i / S; 
            int s = (i * N + j) % (N * S);
            x[p][s] = x_cpu[i * N + j];
            y[p][s] = y_cpu[j * N + i]; // Y is transposed, so the GPU matrix is column-major;
            z[i * N + j] = 0;
        }
    }
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         std::cout << "xcpu[" << i << "][" << j << "]=" << x_cpu[i * N + j] << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         std::cout << "ycpu[" << i << "][" << j << "]=" << y_cpu[i * N + j] << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    // for (int i = 0; i < P; i++) {
    //     for (int j = 0; j < S * N; j++) {
    //         std::cout << "x[" << i << "][" << j << "]=" << x[i][j] << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    // for (int i = 0; i < P; i++) {
    //     for (int j = 0; j < S * N; j++) {
    //         std::cout << "y[" << i << "][" << j << "]=" << y[i][j] << ", ";
    //     }
    //     std::cout << std::endl;
    // }
}

void Benchmark13M::execute_sync(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks, num_blocks);
    if (do_prefetch && pascalGpu) {
        for (int p1 = 0; p1 < P; p1++) {
            for (int p2 = 0; p2 < P; p2++) {
                // int p = p1 * P + p2;
                // cudaMemPrefetchAsync(z[p], sizeof(float) * S * S, 0, 0);
                // Redundant prefetching in the sync implementation, but possibly necessary in multi-GPU;
                cudaMemPrefetchAsync(x[p1], sizeof(float) * S * N, 0, 0);
                cudaMemPrefetchAsync(y[p2], sizeof(float) * S * N, 0, 0);
                cudaDeviceSynchronize();
            }
        }
    }
    cudaDeviceSynchronize();
    for (int p1 = 0; p1 < P; p1++) {
        for (int p2 = 0; p2 < P; p2++) {
#if PARTITION_Z
            matrix_matrix_mult_1<<<grid_size, block_size_2d_dim>>>(x[p1], y[p2], z[p1 * P + p2], std::min(S, N - p1 * S), N, std::min(S, N - p2 * S), S);
#else
            // matrix_matrix_mult_1<<<grid_size, block_size_2d_dim>>>(x[p1], y[p2], z, std::min(S, N - p1 * S), N, std::min(S, N - p2 * S), p1 * S, p2 * S);
            matrix_matrix_mult_2<<<grid_size, block_size_2d_dim, 2 * block_size_2d * block_size_2d * sizeof(float)>>>(x[p1], y[p2], z, std::min(S, N - p1 * S), N, std::min(S, N - p2 * S), N, p1 * S, p2 * S);
#endif
            cudaDeviceSynchronize();
        }
    } 
}

void Benchmark13M::execute_async(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks, num_blocks);

    for (int p1 = 0; p1 < P; p1++) {
#if !P2_STREAMS
        if (pascalGpu && do_prefetch) {  
            cudaSetDevice(select_gpu(p1, max_devices));
            cudaMemPrefetchAsync(x[p1], sizeof(float) * S * N, select_gpu(p1, max_devices), s[p1]);
            if (p1 == 0) cudaMemPrefetchAsync(z, sizeof(float) * N * N, select_gpu(p1, max_devices), s[p1]);
        }   
#endif
        for (int p2 = 0; p2 < P; p2++) {
#if P2_STREAMS
            int p = p1 * P + p2;  
            cudaSetDevice(select_gpu(p, max_devices));
    #if PARTITION_Z
            matrix_matrix_mult_1<<<grid_size, block_size_2d_dim, 0, s[p]>>>(x[p1], y[p2], z[p], std::min(S, N - p1 * S), N, std::min(S, N - p2 * S), S);
    #else
            matrix_matrix_mult_2<<<grid_size, block_size_2d_dim, 2 * block_size_2d * block_size_2d * sizeof(float), s[p]>>>(x[p1], y[p2], z, std::min(S, N - p1 * S), N, std::min(S, N - p2 * S), N, p1 * S, p2 * S);
    #endif
#else
            if (pascalGpu && do_prefetch && (p1 == 0)) {  
                cudaSetDevice(select_gpu(p1, max_devices));
                cudaMemPrefetchAsync(y[p2], sizeof(float) * S * N, select_gpu(p1, max_devices), s[p2]);
            } 
            cudaSetDevice(select_gpu(p1, max_devices));
            matrix_matrix_mult_2<<<grid_size, block_size_2d_dim, 2 * block_size_2d * block_size_2d * sizeof(float), s[p1]>>>(x[p1], y[p2], z, std::min(S, N - p1 * S), N, std::min(S, N - p2 * S), N, p1 * S, p2 * S);
#endif
        }
    }
    // Synchronization;
    for (int p1 = 0; p1 < P; p1++) {
#if P2_STREAMS
        for (int p2 = 0; p2 < P; p2++) {
            int p = p1 * P + p2;  
            err = cudaStreamSynchronize(s[p]);
        }
#else
        err = cudaStreamSynchronize(s[p1]);
#endif
    }
}

std::string Benchmark13M::print_result(bool short_form) {
    if (short_form) {
#if PARTITION_Z
        return std::to_string(z[0][0]);
#else
        return std::to_string(z[0]);
#endif
    } else {
        int old_precision = std::cout.precision();
		std::cout.precision(2);
        std::string res = "[\n";
        for (int i = 0; i < std::min(30, N); i++) {
            res += "[";
            for (int j = 0; j < std::min(30, N); j++) {
#if PARTITION_Z
                int p1 = i / S; 
                int p2 = j / S; 
                res += std::to_string(z[p1 * P + p2][(i % S) * S + j % S]) + ", ";
#else
                res += std::to_string(z[i * N + j]) + ", ";
#endif
            }
            res += "...]\n";
        }
        std::cout.precision(old_precision);
        return res + "...]";
    }
}

void Benchmark13M::cpu_validation(int iter) {
#if CPU_VALIDATION
    if (iter > 0) return;
    int max_errors = 20;
    int errors = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float res = 0;
            for (int k = 0; k < N; k++) {
                res += x_cpu[i * N + k] * y_cpu[k * N + j];
            }
            z_cpu[i * N + j] = res;
#if !PARTITION_Z
            if (std::abs(z[i * N + j] - z_cpu[i * N + j]) > 1e-3) {
                if (errors < max_errors) std::cout << "error, z[" << i << "][" << j << "]=" << z[i * N + j] << ", cpu=" << z_cpu[i * N + j] << std::endl;
                errors += 1;
            }
#endif
        }
    }
    std::cout << "errors=" << errors << std::endl;
#endif
}