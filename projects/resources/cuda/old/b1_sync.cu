#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include "utils.hpp"
#include "options.hpp"
#include "b1.cuh"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

void init(float *x, float *y, int N) {
    for (int i = 0; i < N; i++) {
        x[i] = 1.0 / (i + 1);
        y[i] = 2.0 / (i + 1);
    }
}

void reset(float *res, float *x, float *y, int N) {
    for (int i = 0; i < N; i++) {
        x[i] = 1.0 / (i + 1);
        y[i] = 2.0 / (i + 1);
    }
    res[0] = 0.0;
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

    if (debug) {
        std::cout << "running b1 sync" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();
	float *x, *y, *x1, *y1, *res;
    err = cudaMallocManaged(&x, sizeof(float) * N);
    err = cudaMallocManaged(&y, sizeof(float) * N);
    err = cudaMallocManaged(&x1, sizeof(float) * N);
    err = cudaMallocManaged(&y1, sizeof(float) * N);
    err = cudaMallocManaged(&res, sizeof(float));
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;
    init(x, y, N);

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset(res, x, y, N);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();

        square<<<num_blocks, block_size>>>(x, x1, N);
        err = cudaDeviceSynchronize();
        square<<<num_blocks, block_size>>>(y, y1, N);
        err = cudaDeviceSynchronize();
        reduce<<<num_blocks, block_size>>>(x1, y1, res, N);        
        err = cudaDeviceSynchronize();
        if (debug && err) std::cout << err << std::endl;

        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        if (i >= skip_iterations) tot += tmp;

        if (debug) {
            std::cout << "  gpu result=" << res[0] << "; time=" << (float) tmp / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << res[0] << "," << (float) (reset_time + tmp) / 1e6 << "," << (float) reset_time / 1e6 << "," << (float) tmp / 1e6 << std::endl;
        }
    }

    // Print;
	cudaDeviceSynchronize();
    
    if (debug) std::cout << "\nmean exec time=" << (float) tot / (1000 * (num_executions - skip_iterations)) << " ms" << std::endl;
}
