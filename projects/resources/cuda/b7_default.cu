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
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] / val[0];
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
}

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

    srand(time(0));

    Options options = Options(argc, argv);
	int debug = options.debug;

    int num_executions = options.num_iter;
    int N = options.N;

    int degree = 50;
    int iterations = 10;

    int block_size = options.block_size_1d;
    int num_blocks = 32;
    int err = 0;

    int nnz = degree * N;

    if (debug) {
        std::cout << "running b7 sync" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size << std::endl;
    }
    
    auto start = clock_type::now();
    int *ptr, *idx, *val, *ptr2, *idx2, *val2;
    float *auth1, *auth2, *hub1, *hub2, *auth_norm, *hub_norm;

    err = cudaMallocManaged(&ptr, sizeof(int) * (N + 1));
    err = cudaMallocManaged(&ptr2, sizeof(int) * (N + 1));
    err = cudaMallocManaged(&idx, sizeof(int) * nnz);
    err = cudaMallocManaged(&idx2, sizeof(int) * nnz);
    err = cudaMallocManaged(&val, sizeof(int) * nnz);
    err = cudaMallocManaged(&val2, sizeof(int) * nnz);

    err = cudaMallocManaged(&auth1, sizeof(float) * N);
    err = cudaMallocManaged(&auth2, sizeof(float) * N);
    err = cudaMallocManaged(&hub1, sizeof(float) * N);
    err = cudaMallocManaged(&hub2, sizeof(float) * N);
    err = cudaMallocManaged(&auth_norm, sizeof(float));
    err = cudaMallocManaged(&hub_norm, sizeof(float));
   
    if (debug && err) std::cout << err << std::endl;

    // Create streams;
    cudaStream_t s1, s2;
    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;

    // Create a random COO;
    int *x = (int*) malloc(nnz * sizeof(int));
    int *y = (int*) malloc(nnz * sizeof(int));
    int *v = (int*) malloc(nnz * sizeof(int));
    random_coo(x, y, v, N, degree);

    // Create a CSR;
    coo2csr(ptr, idx, val, x, y, v, N, N, nnz);
    coo2csr(ptr2, idx2, val2, y, x, v, N, N, nnz);

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset(auth1, auth2, hub1, hub2, auth_norm, hub_norm, N);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();

        for (int iter = 0; iter < iterations; iter++) {
            cudaEvent_t e1, e2;
            cudaEventCreate(&e1);
            cudaEventCreate(&e2);

            spmv<<<num_blocks, block_size, 0, s1>>>(ptr2, idx2, val2, hub1, auth2, N, nnz);
            err = cudaEventRecord(e1, s1);
            spmv<<<num_blocks, block_size, 0, s2>>>(ptr, idx, val, auth1, hub2, N, nnz);
            err = cudaEventRecord(e2, s2);
            sum<<<num_blocks, block_size, 0, s1>>>(auth2, auth_norm, N);

            sum<<<num_blocks, block_size, 0, s2>>>(hub2, hub_norm, N);

            // Stream 1 waits stream 2;
            err = cudaStreamWaitEvent(s1, e2, 0);
            divide<<<num_blocks, block_size, 0, s1>>>(auth2, auth1, auth_norm, N);
            // Stream 2 waits stream 1;
            err = cudaStreamWaitEvent(s2, e1, 0);
            divide<<<num_blocks, block_size, 0, s2>>>(hub2, hub1, hub_norm, N);

            err = cudaStreamSynchronize(s1);
            err = cudaStreamSynchronize(s2);
            auth_norm[0] = 0;
            
            hub_norm[0] = 0;

            if (debug && err) std::cout << err << std::endl;
        }

        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        tot += tmp;

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
    
    if (debug) std::cout << "\nmean exec time=" << (float) tot / (1000 * num_executions) << " ms" << std::endl;
}
