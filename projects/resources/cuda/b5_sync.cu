#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include "utils.hpp"
#include "options.hpp"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

extern "C" __global__ void sum(int* x, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] += 1;
    }
}

/////////////////////////////
/////////////////////////////

void init(int **x, int N, int K) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            x[i][j] = rand() % 10;
        }
    }
}

void reset() {
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
    int num_blocks = 64;
    int err = 0;

    int K = num_executions;

    if (debug) {
        std::cout << "running b5 sync" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size << std::endl;
    }
    
    auto start = clock_type::now();
    int **x;
    x = (int **) malloc(sizeof(int*) * K);
    for (int i = 0; i < K; i++) {
        cudaMallocManaged(&x[i], sizeof(int) * N);
    }
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;
    init(x, N, K);

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset();
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();

        sum<<<num_blocks, block_size>>>(x[i], N);
        err = cudaDeviceSynchronize();

        if (debug && err) std::cout << err << std::endl;

        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        tot += tmp;

        if (debug) {
            std::cout << "  gpu result=[";
            for (int j = 0; j < K; j++) {
                std::cout << x[j][0] << ", ";
            } 
            std::cout << ", ...]; time=" << (float) tmp / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << x[0][0] << "," << (float) (reset_time + tmp) / 1e6 << "," << (float) reset_time / 1e6 << "," << (float) tmp / 1e6 << std::endl;
        }
    }

    // Print;
	cudaDeviceSynchronize();
    
    if (debug) std::cout << "\nmean exec time=" << (float) tot / (1000 * num_executions) << " ms" << std::endl;
}
