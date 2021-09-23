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
#include <set>
#include "utils.hpp"
#include "options.hpp"
#include <vector>
#include <tuple>

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

#define NUM_THREADS_PER_BLOCK 32
#define THREADS_PER_VECTOR 4
#define MAX_NUM_VECTORS_PER_BLOCK (1024 / THREADS_PER_VECTOR)

extern "C" __global__ void spmv(const int *ptr, const int *idx, const int *val, const float *vec, float *res, int num_rows, int num_nnz) {

    for(int n = blockIdx.x * blockDim.x + threadIdx.x; n < num_rows; n += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int i = ptr[n]; i < ptr[n + 1]; i++) {
            sum += val[i] * vec[idx[i]];
        }
        res[n] = sum;
    }
}

extern "C" __global__ void spmv2(int* cudaRowCounter, int* d_ptr, int* d_cols, float* d_val, float* d_vector, float* d_out, int N) {
    int i;
    int thread_per_vector = THREADS_PER_VECTOR;
    float sum;
    int row;
    int rowStart, rowEnd;
    int laneId = threadIdx.x % thread_per_vector; //lane index in the vector
    int vectorId = threadIdx.x / thread_per_vector; //vector index in the thread block
    int warpLaneId = threadIdx.x & 31;	//lane index in the warp
    int warpVectorId = warpLaneId / thread_per_vector;	//vector index in the warp

    __shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

    // Get the row index
    if (warpLaneId == 0) {
        row = atomicAdd(cudaRowCounter, 32 / thread_per_vector);
    }
    // Broadcast the value to other threads in the same warp and compute the row index of each vector
    row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

    while (row < N) {

        // Use two threads to fetch the row offset
        if (laneId < 2) {
            space[vectorId][laneId] = d_ptr[row + laneId];
        }
        rowStart = space[vectorId][0];
        rowEnd = space[vectorId][1];

        sum = 0;
        // Compute dot product
        if (thread_per_vector == 32) {

            // Ensure aligned memory access
            i = rowStart - (rowStart & (thread_per_vector - 1)) + laneId;

            // Process the unaligned part
            if (i >= rowStart && i < rowEnd) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }

                // Process the aligned part
            for (i += thread_per_vector; i < rowEnd; i += thread_per_vector) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        } else {
            for (i = rowStart + laneId; i < rowEnd; i += thread_per_vector) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        }
        // Intra-vector reduction
        for (i = thread_per_vector >> 1; i > 0; i >>= 1) {
            sum += __shfl_down_sync(0xffffffff,sum, i);
        }

        // Save the results
        if (laneId == 0) {
            d_out[row] = sum;
        }

        // Get a new row index
        if(warpLaneId == 0) {
            row = atomicAdd(cudaRowCounter, 32 / thread_per_vector);
        }
        // Broadcast the row index to the other threads in the same warp and compute the row index of each vector
        row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;
    }
}

// Compute d_out = y + alpha * A * d_vector;
extern "C" __global__ void spmv_full(int* cudaRowCounter, int* d_ptr, int* d_cols, float* d_val, float* d_vector, float* d_out, int N, float alpha, float* y) {
    int i;
    int thread_per_vector = THREADS_PER_VECTOR;
    float sum;
    int row;
    int rowStart, rowEnd;
    int laneId = threadIdx.x % thread_per_vector; //lane index in the vector
    int vectorId = threadIdx.x / thread_per_vector; //vector index in the thread block
    int warpLaneId = threadIdx.x & 31;	//lane index in the warp
    int warpVectorId = warpLaneId / thread_per_vector;	//vector index in the warp

    __shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

    // Get the row index
    if (warpLaneId == 0) {
        row = atomicAdd(cudaRowCounter, 32 / thread_per_vector);
    }
    // Broadcast the value to other threads in the same warp and compute the row index of each vector
    row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

    while (row < N) {

        // Use two threads to fetch the row offset
        if (laneId < 2) {
            space[vectorId][laneId] = d_ptr[row + laneId];
        }
        rowStart = space[vectorId][0];
        rowEnd = space[vectorId][1];

        sum = 0;
        // Compute dot product
        if (thread_per_vector == 32) {

            // Ensure aligned memory access
            i = rowStart - (rowStart & (thread_per_vector - 1)) + laneId;

            // Process the unaligned part
            if (i >= rowStart && i < rowEnd) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }

                // Process the aligned part
            for (i += thread_per_vector; i < rowEnd; i += thread_per_vector) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        } else {
            for (i = rowStart + laneId; i < rowEnd; i += thread_per_vector) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        }
        // Intra-vector reduction
        for (i = thread_per_vector >> 1; i > 0; i >>= 1) {
            sum += __shfl_down_sync(0xffffffff,sum, i);
        }

        // Save the results
        if (laneId == 0) {
            d_out[row] = y[row] + alpha * sum;
        }

        // Get a new row index
        if(warpLaneId == 0) {
            row = atomicAdd(cudaRowCounter, 32 / thread_per_vector);
        }
        // Broadcast the row index to the other threads in the same warp and compute the row index of each vector
        row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void vector_norm(const float *x, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] * x[i];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
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

// Compute y = val + alpha * x;
extern "C" __global__ void saxpy(float* y, float *val, float *x, float alpha, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = val[i] + alpha * x[i];
    }
}

extern "C" __global__ void cpy(float *y, const float *x, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i];
    }
}

/////////////////////////////
/////////////////////////////

void reset(float *x, float *t1, float *t2, int *row_cnt_1, int *row_cnt_2, int N) {
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
    }
    *t1 = 0.0;
    *t2 = 0.0;
    *row_cnt_1 = 0;
    *row_cnt_2 = 0;
}

void init(float *b, int N) {
    for (int i = 0; i < N; i++) {
        b[i] = (float)(rand()) / (float)(RAND_MAX);
    }
}

template<typename I, typename T>
void random_coo(I* x, I *y, T *val, int N, int degree) {
    // Create random matrix entries;
    std::vector<std::tuple<I, I, T>> t;
    for (int i = 0; i < N; i++) {
        std::set<I> edges;
        while (edges.size() < degree) {
            I edge = (I) rand() % N;
            if (i != edge) {
                edges.insert(edge);
            }
        }
        for (auto e = edges.begin(); e != edges.end(); e++) {
            T tmp = (T)(rand()) / (T)(RAND_MAX);
            auto tuple1 = std::make_tuple(i, *e, tmp);
            auto tuple2 = std::make_tuple(*e, i, tmp);
            t.push_back(tuple1);
            t.push_back(tuple2);
        }
    }
    int i = 0;
    for (auto t_i = t.begin(); t_i != t.end(); t_i++, i++) {
        x[i] = std::get<0>(*t_i);
        y[i] = std::get<1>(*t_i);
        val[i] = std::get<2>(*t_i);
    }
}


/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

    srand(time(0));

    Options options = Options(argc, argv);
	int debug = options.debug;

    int num_executions = options.num_iter;
    int N = options.N;

    int max_degree = 2;
    int iterations = 10;

    int num_blocks = options.num_blocks;
    int block_size_1d = options.block_size_1d;
    int block_size_2d = options.block_size_2d;
    int skip_iterations = options.skip_iterations;
    int err = 0;

    if (debug) {
        std::cout << "running b9 sync" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size_1d << std::endl;
        std::cout << "block size 2d=" << block_size_2d << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();

    int nnz = N * max_degree * 2;

    int *ptr, *idx, *rowCounter1, *rowCounter2;
    float *x, *b, *val, *p, *r, *t1, *t2, *y;

    err = cudaMallocManaged(&ptr, sizeof(int) * (N + 1));
    err = cudaMallocManaged(&idx, sizeof(int) * nnz);
    err = cudaMallocManaged(&val, sizeof(float) * nnz);
    err = cudaMallocManaged(&rowCounter1, sizeof(int));
    err = cudaMallocManaged(&rowCounter2, sizeof(int));

    err = cudaMallocManaged(&x, sizeof(float) * N);
    err = cudaMallocManaged(&b, sizeof(float) * N);
    err = cudaMallocManaged(&p, sizeof(float) * N);
    err = cudaMallocManaged(&r, sizeof(float) * N);
    err = cudaMallocManaged(&y, sizeof(float) * N);
    err = cudaMallocManaged(&t1, sizeof(float));
    err = cudaMallocManaged(&t2, sizeof(float));
   
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;
    start = clock_type::now();

    int *x_coo = (int*) malloc(nnz * sizeof(int));
    int *y_coo = (int*) malloc(nnz * sizeof(int));
    float *v_coo = (float*) malloc(nnz * sizeof(float));
    random_coo(x_coo, y_coo, v_coo, N, max_degree);
    coo2csr(ptr, idx, val, x_coo, y_coo, v_coo, N, N, nnz);
    init(b, N);

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset(x, t1, t2, rowCounter1, rowCounter2, N);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        int nb = ceil(N / ((float) block_size_1d));

        start = clock_type::now();

        spmv_full<<<nb, block_size_1d, block_size_1d * sizeof(float)>>>(rowCounter1, ptr, idx, val, x, r, N, -1.0, b);
        // err = cudaDeviceSynchronize();
        cpy<<<num_blocks, block_size_1d>>>(p, r, N);
        // err = cudaDeviceSynchronize();
        vector_norm<<<num_blocks, block_size_1d>>>(r, t1, N);
        // err = cudaDeviceSynchronize();

        for (int iter = 0; iter < iterations; iter++) {
            spmv2<<<nb, block_size_1d, block_size_1d * sizeof(float)>>>(rowCounter2, ptr, idx, val, p, y, N);
            // err = cudaDeviceSynchronize();
            dot_product<<<num_blocks, block_size_1d>>>(p, y, t2, N);
            err = cudaDeviceSynchronize();
            float alpha = *t1 / *t2;
            float old_t1 = *t1;
            *t1 = 0.0;
            *rowCounter1 = 0;
            *rowCounter2 = 0;
            saxpy<<<num_blocks, block_size_1d>>>(x, x, p, alpha, N);
            // err = cudaDeviceSynchronize();
            saxpy<<<num_blocks, block_size_1d>>>(r, r, y, -1.0 * alpha, N);
            // err = cudaDeviceSynchronize();
            vector_norm<<<num_blocks, block_size_1d>>>(r, t1, N);
            err = cudaDeviceSynchronize();
            float beta = *t1 / old_t1;
            saxpy<<<num_blocks, block_size_1d>>>(p, r, p, beta, N);
            // err = cudaDeviceSynchronize();
        }
        err = cudaDeviceSynchronize();
        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        tot += tmp;

        if (debug) {
            std::cout << "  gpu result=[";
            for (int j = 0; j < 10; j++) {
                std::cout << x[j] << ", ";
            } 
            std::cout << ", ...]; time=" << (float) tmp / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << 0.0 << "," << (float) (reset_time + tmp) / 1e6 << "," << (float) reset_time / 1e6 << "," << (float) tmp / 1e6 << std::endl;
        }
    }

    // Print;
	cudaDeviceSynchronize();
    
    if (debug) std::cout << "\nmean exec time=" << (float) tot / (1000 * num_executions) << " ms" << std::endl;
}
