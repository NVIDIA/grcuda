#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include <set>
#include "utils.hpp"
#include "options.hpp"
#include "b7.cuh"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

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
    int num_blocks = options.num_blocks;
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
