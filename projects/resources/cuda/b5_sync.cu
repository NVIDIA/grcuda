#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include "utils.hpp"
#include "options.hpp"
#include "b5.cuh"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

void init(double **x, double **y, double* tmp_x, int N, int K) {
    for (int j = 0; j < N; j++) {
        tmp_x[j] = 60 - 0.5 + (double) rand() / RAND_MAX;
        for (int i = 0; i < K; i++) {
            x[i][j] = tmp_x[j];
            // y[i][j] = 0;
        }
    }
}

void reset(double **x, double* y, int N, int K) {
    for (int i = 0; i < K; i++) {
        // memcpy(x[i], y, sizeof(int) * N);
        // cudaMemcpy(x[i], y, sizeof(double) * N, cudaMemcpyDefault);
        // cudaMemcpyAsync(x[i], y, sizeof(int) * N, cudaMemcpyHostToDevice, s[i]);
        for (int j = 0; j < N; j++) {
            x[i][j] = y[j];
        }
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

    int block_size = options.block_size_1d;
    int num_blocks = options.num_blocks;
    int skip_iterations = options.skip_iterations;
    int err = 0;

    int M = 10;

    if (debug) {
        std::cout << "running b5 sync" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();
    double **x = (double **) malloc(sizeof(double*) * M);
    double **y = (double **) malloc(sizeof(double*) * M);
    double *tmp_x = (double *) malloc(sizeof(double) * N);
    cudaHostRegister(tmp_x, sizeof(double) * N, 0);

    for (int i = 0; i < M; i++) {
        cudaMallocManaged(&x[i], sizeof(double) * N);
        cudaMallocManaged(&y[i], sizeof(double) * N);
    }
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;
    init(x, y, tmp_x, N, M);

    if (debug) std::cout << "x[0][0]=" << tmp_x[0] << std::endl;

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (double) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    double tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset(x, tmp_x, N, M);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (double) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();
        for (int j = 0; j < M; j++) {
            bs<<<num_blocks, block_size>>>(x[j], y[j], N, R, V, T, K);
            err = cudaDeviceSynchronize();
        }

        if (debug && err) std::cout << err << std::endl;

        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        if (i >= skip_iterations) tot += tmp;

        if (debug) {
            std::cout << "  gpu result=[";
            for (int j = 0; j < M; j++) {
                std::cout << y[j][0] << ", ";
            } 
            std::cout << ", ...]; time=" << (double) tmp / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << y[0][0] << "," << (double) (reset_time + tmp) / 1e6 << "," << (double) reset_time / 1e6 << "," << (double) tmp / 1e6 << std::endl;
        }
    }

    // Print;
	cudaDeviceSynchronize();
    
    if (debug) std::cout << "\nmean exec time=" << (double) tot / (1000 * (num_executions - skip_iterations)) << " ms" << std::endl;
}
