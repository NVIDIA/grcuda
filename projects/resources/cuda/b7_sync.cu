#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include <set>
#include "utils.hpp"
#include "options.hpp"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

#define WARP_SIZE 32
#define THREADS_PER_VECTOR 4
#define MAX_NUM_VECTORS_PER_BLOCK (1024 / THREADS_PER_VECTOR)

/////////////////////////////
/////////////////////////////

extern "C" __global__ void spmv(const int *ptr, const int *idx, const int *val, const float *vec, float *res, int num_rows, int num_nnz) {

    for(int n = blockIdx.x * blockDim.x + threadIdx.x; n < num_rows; n += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int i = ptr[n]; i < ptr[n + 1]; i++) {
            sum += val[i] * vec[idx[i]];
        }
        res[n] = sum;
    }
}


extern "C" __global__ void spmv2(const int *ptr, const int *idx, const int *val, const float *vec, float *res, int num_rows, int num_nnz) {
    // Thread ID in block
    int t = threadIdx.x;

    // Thread ID in warp
    int lane = t & (WARP_SIZE - 1);

    // Number of warps per block
    int warpsPerBlock = blockDim.x / WARP_SIZE;

    // One row per warp
    int row = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);

    extern __shared__ volatile float vals[];

    if (row < num_rows) {
        int rowStart = ptr[row];
        int rowEnd = ptr[row+1];
        float sum = 0;

        // Use all threads in a warp accumulate multiplied elements
        for (int j = rowStart + lane; j < rowEnd; j += WARP_SIZE) {
            int col = idx[j];
            sum += val[j] * vec[col];
        }
        vals[t] = sum;
        __syncthreads();

        // Reduce partial sums
        if (lane < 16) vals[t] += vals[t + 16];
        if (lane <  8) vals[t] += vals[t + 8];
        if (lane <  4) vals[t] += vals[t + 4];
        if (lane <  2) vals[t] += vals[t + 2];
        if (lane <  1) vals[t] += vals[t + 1];
        __syncthreads();

        // Write result
        if (lane == 0) {
            res[row] = vals[t];
        }
    }	
}

extern "C" __global__ void spmv3(int* cudaRowCounter, int* d_ptr, int* d_cols, int* d_val, float* d_vector, float* d_out, int N) {
	int i;
	float sum;
	int row;
	int rowStart, rowEnd;
	int laneId = threadIdx.x % THREADS_PER_VECTOR; //lane index in the vector
	int vectorId = threadIdx.x / THREADS_PER_VECTOR; //vector index in the thread block
	int warpLaneId = threadIdx.x & 31;	//lane index in the warp
	int warpVectorId = warpLaneId / THREADS_PER_VECTOR;	//vector index in the warp

	__shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

	// Get the row index
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
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
		if (THREADS_PER_VECTOR == 32) {

			// Ensure aligned memory access
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			// Process the unaligned part
			if (i >= rowStart && i < rowEnd) {
				sum += d_val[i] * d_vector[d_cols[i]];
			}

				// Process the aligned part
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += d_val[i] * d_vector[d_cols[i]];
			}
		} else {
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += d_val[i] * d_vector[d_cols[i]];
			}
		}
		// Intra-vector reduction
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
            sum += __shfl_down_sync(0xffffffff,sum, i);
		}

		// Save the results
		if (laneId == 0) {
			d_out[row] = sum;
		}

		// Get a new row index
		if(warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
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

extern "C" __global__ void sum(const float *x, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

extern "C" __global__ void divide(const float* x, float *y, float *val, int n) {
    float v = val[0];
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] / v;
    }
}

/////////////////////////////
/////////////////////////////

void random_coo(int* x, int *y, int *val, int N, int degree) {
    for (int i = 0; i < N; i++) {
        std::set<int> edges;
        while (edges.size() < degree) {
            edges.insert(rand() % N);
        }
        int j = 0;
        for (auto iter = edges.begin(); iter != edges.end(); iter++, j++) {
            x[i * degree + j] = i;
            y[i * degree + j] = *iter;
            val[i * degree + j] = 1;
        }
    }
}

void reset(float *auth1, float *auth2, float *hub1, float* hub2, float *auth_norm, float *hub_norm, int N) {
    for (int i = 0; i < N; i++) {
        auth1[i] = 1;
        auth2[i] = 1;
        hub1[i] = 1;
        hub2[i] = 1;
    }
    auth_norm[0] = 0;
    hub_norm[0] = 0;

    // cudaMemPrefetchAsync(auth1, N * sizeof(float), 0);
    // cudaMemPrefetchAsync(auth2, N * sizeof(float), 0);
    // cudaMemPrefetchAsync(hub1, N * sizeof(float), 0);
    // cudaMemPrefetchAsync(hub2, N * sizeof(float), 0);
    // cudaMemPrefetchAsync(auth_norm, sizeof(float), 0);
    // cudaMemPrefetchAsync(hub_norm, sizeof(float), 0);
    // cudaDeviceSynchronize();
}

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

    srand(time(0));

    Options options = Options(argc, argv);
	int debug = options.debug;

    int num_executions = options.num_iter;
    int N = options.N;

    int degree = 3;
    int iterations = 5;

    int block_size = options.block_size_1d;
    int num_blocks = 32; // options.num_blocks;
    int skip_iterations = options.skip_iterations;
    int err = 0;

    int nnz = degree * N;

    if (debug) {
        std::cout << "running b7 sync" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();
    int *ptr, *idx, *val, *ptr2, *idx2, *val2, *rowCounter1, *rowCounter2;
    int *ptr_tmp, *idx_tmp, *val_tmp, *ptr2_tmp, *idx2_tmp, *val2_tmp;

    float *auth1, *auth2, *hub1, *hub2, *auth_norm, *hub_norm;

    // Use temporary CPU vectors to simplify reinitialization at each benchmark execution;
    ptr_tmp = (int *) malloc(sizeof(int) * (N + 1));
    ptr2_tmp = (int *) malloc(sizeof(int) * (N + 1));
    idx_tmp = (int *) malloc(sizeof(int) * nnz);
    idx2_tmp = (int *) malloc(sizeof(int) * nnz);
    val_tmp = (int *) malloc(sizeof(int) * nnz);
    val2_tmp = (int *) malloc(sizeof(int) * nnz);

    err = cudaMallocManaged(&ptr, sizeof(int) * (N + 1));
    err = cudaMallocManaged(&ptr2, sizeof(int) * (N + 1));
    err = cudaMallocManaged(&idx, sizeof(int) * nnz);
    err = cudaMallocManaged(&idx2, sizeof(int) * nnz);
    err = cudaMallocManaged(&val, sizeof(int) * nnz);
    err = cudaMallocManaged(&val2, sizeof(int) * nnz);
    err = cudaMallocManaged(&rowCounter1, sizeof(int));
    err = cudaMallocManaged(&rowCounter2, sizeof(int));

    err = cudaMallocManaged(&auth1, sizeof(float) * N);
    err = cudaMallocManaged(&auth2, sizeof(float) * N);
    err = cudaMallocManaged(&hub1, sizeof(float) * N);
    err = cudaMallocManaged(&hub2, sizeof(float) * N);
    err = cudaMallocManaged(&auth_norm, sizeof(float));
    err = cudaMallocManaged(&hub_norm, sizeof(float));
   
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;

    // Create a random COO;
    int *x = (int*) malloc(nnz * sizeof(int));
    int *y = (int*) malloc(nnz * sizeof(int));
    int *v = (int*) malloc(nnz * sizeof(int));
    random_coo(x, y, v, N, degree);

    // Create a CSR;
    coo2csr(ptr_tmp, idx_tmp, val_tmp, x, y, v, N, N, nnz);
    coo2csr(ptr2_tmp, idx2_tmp, val2_tmp, y, x, v, N, N, nnz);

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {

        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        for (int j = 0; j < nnz; j++) {
            idx[j] = idx_tmp[j];
            idx2[j] = idx2_tmp[j];
            val[j] = val_tmp[j];
            val2[j] = val2_tmp[j];
        }
        for (int j = 0; j < N + 1; j++) {
            ptr[j] = ptr_tmp[j];
            ptr2[j] = ptr2_tmp[j];
        }
        reset(auth1, auth2, hub1, hub2, auth_norm, hub_norm, N);
        rowCounter1[0] = 0;
        rowCounter2[0] = 0;
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();

        for (int iter = 0; iter < iterations; iter++) {

            // cudaMemPrefetchAsync(auth1, N * sizeof(float), 0);
            // cudaMemPrefetchAsync(auth2, N * sizeof(float), 0);
            // cudaMemPrefetchAsync(hub1, N * sizeof(float), 0);
            // cudaMemPrefetchAsync(hub2, N * sizeof(float), 0);
            // cudaMemPrefetchAsync(auth_norm, sizeof(float), 0);
            // cudaMemPrefetchAsync(hub_norm, sizeof(float), 0);
            // cudaDeviceSynchronize();

            int nb = ceil(N / ((float) block_size));

            // spmv<<<nb, block_size>>>(ptr2, idx2, val2, hub1, auth2, N, nnz);
            spmv3<<<nb, block_size, block_size * sizeof(float)>>>(rowCounter1, ptr2, idx2, val2, hub1, auth2, N);
            err = cudaDeviceSynchronize();

            // spmv<<<nb, block_size>>>(ptr, idx, val, auth1, hub2, N, nnz);
            spmv3<<<nb, block_size, block_size * sizeof(float)>>>(rowCounter2, ptr, idx, val, auth1, hub2, N);
            err = cudaDeviceSynchronize();

            sum<<<num_blocks, block_size>>>(auth2, auth_norm, N);
            err = cudaDeviceSynchronize();

            sum<<<num_blocks, block_size>>>(hub2, hub_norm, N);
            err = cudaDeviceSynchronize();

            divide<<<num_blocks, block_size>>>(auth2, auth1, auth_norm, N);
            err = cudaDeviceSynchronize();

            divide<<<num_blocks, block_size>>>(hub2, hub1, hub_norm, N);
            err = cudaDeviceSynchronize();

            auth_norm[0] = 0;
            hub_norm[0] = 0;
            rowCounter1[0] = 0;
            rowCounter2[0] = 0;

            if (debug && err) std::cout << err << std::endl;
        }

        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        if (i >= skip_iterations) tot += tmp;

        if (debug) {
            std::cout << "  gpu result=[";
            for (int j = 0; j < 10; j++) {
                std::cout << auth1[j] << ", ";
            } 
            std::cout << ", ...]; time=" << (float) tmp / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << 0.0 << "," << (float) (reset_time + tmp) / 1e6 << "," << (float) reset_time / 1e6 << "," << (float) tmp / 1e6 << std::endl;
        }
    }

    // Print;
	cudaDeviceSynchronize();
    
    if (debug) std::cout << "\nmean exec time=" << (float) tot / (1000 * (num_executions - skip_iterations)) << " ms" << std::endl;
}
