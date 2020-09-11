#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include <set>
#include "utils.hpp"
#include "options.hpp"
#include "b8.cuh"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

void reset(float *image, float *maximum, float *minimum, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            image[i * N + j] = 0;
        }
    }
    *maximum = 0;
    *minimum = 0;
}

void gaussian_kernel(float* kernel, int diameter, float sigma) {
    int mean = diameter / 2;
    float sum_tmp = 0;
    for (int i = 0; i < diameter; i++) {
        for (int j = 0; j < diameter; j++) {
            kernel[i * diameter + j] = exp(-0.5 * ((i - mean) * (i - mean) + (j - mean) * (j - mean)) / (sigma * sigma));
            sum_tmp += kernel[i * diameter + j];
        } 
    }
    for (int i = 0; i < diameter; i++) {
        for (int j = 0; j < diameter; j++) {
            kernel[i * diameter + j] /= sum_tmp;
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

    int kernel_small_diameter = 3;
    int kernel_large_diameter = 5;
    int kernel_unsharpen_diameter = 3;

    int block_size_1d = options.block_size_1d;
    int block_size_2d = options.block_size_2d;
    int num_blocks = options.num_blocks;
    int skip_iterations = options.skip_iterations;
    int err = 0;

    if (debug) {
        std::cout << "running b8 sync" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size_1d << std::endl;
        std::cout << "block size 2d=" << block_size_2d << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();
    float *image, *image2, *image3, *image_unsharpen, *mask_small, *mask_large, *mask_unsharpen, *blurred_small, *blurred_large, *blurred_unsharpen;
    float *kernel_small, *kernel_large, *kernel_unsharpen, *maximum, *minimum;
    err = cudaMallocManaged(&image, sizeof(float) * N * N);
    err = cudaMallocManaged(&image2, sizeof(float) * N * N);
    err = cudaMallocManaged(&image3, sizeof(float) * N * N);
    err = cudaMallocManaged(&image_unsharpen, sizeof(float) * N * N);
    err = cudaMallocManaged(&mask_small, sizeof(float) * N * N);
    err = cudaMallocManaged(&mask_large, sizeof(float) * N * N);
    err = cudaMallocManaged(&mask_unsharpen, sizeof(float) * N * N);
    err = cudaMallocManaged(&blurred_small, sizeof(float) * N * N);
    err = cudaMallocManaged(&blurred_large, sizeof(float) * N * N);
    err = cudaMallocManaged(&blurred_unsharpen, sizeof(float) * N * N);

    err = cudaMallocManaged(&kernel_small, sizeof(float) * kernel_small_diameter * kernel_small_diameter);
    err = cudaMallocManaged(&kernel_large, sizeof(float) * kernel_large_diameter * kernel_large_diameter);
    err = cudaMallocManaged(&kernel_unsharpen, sizeof(float) * kernel_unsharpen_diameter * kernel_unsharpen_diameter);
    err = cudaMallocManaged(&maximum, sizeof(float));
    err = cudaMallocManaged(&minimum, sizeof(float));
   
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;
    start = clock_type::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            image[i * N + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    gaussian_kernel(kernel_small, kernel_small_diameter, 1);
    gaussian_kernel(kernel_large, kernel_large_diameter, 10);
    gaussian_kernel(kernel_unsharpen, kernel_unsharpen_diameter, 5);

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset(image3, maximum, minimum, N);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();

        dim3 block_size_2d_dim(block_size_2d, block_size_2d);
        dim3 grid_size(num_blocks, num_blocks);
        dim3 grid_size_2(num_blocks / 2, num_blocks / 2);

        gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_small_diameter * kernel_small_diameter * sizeof(float)>>>(image, blurred_small, N, N, kernel_small, kernel_small_diameter);
        cudaDeviceSynchronize();

        gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_large_diameter * kernel_large_diameter * sizeof(float)>>>(image, blurred_large, N, N, kernel_large, kernel_large_diameter);
        cudaDeviceSynchronize();

        gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_unsharpen_diameter * kernel_unsharpen_diameter * sizeof(float)>>>(image, blurred_unsharpen, N, N, kernel_unsharpen, kernel_unsharpen_diameter);
        cudaDeviceSynchronize();

        sobel<<<grid_size_2, block_size_2d_dim>>>(blurred_small, mask_small, N, N);
        cudaDeviceSynchronize();

        sobel<<<grid_size_2, block_size_2d_dim>>>(blurred_large, mask_large, N, N);
        cudaDeviceSynchronize();

        maximum_kernel<<<num_blocks, block_size_1d>>>(mask_large, maximum, N * N);
    	cudaDeviceSynchronize();

        minimum_kernel<<<num_blocks, block_size_1d>>>(mask_large, minimum, N * N);
        cudaDeviceSynchronize();

        extend<<<num_blocks, block_size_1d>>>(mask_large, minimum, maximum, N * N);
        cudaDeviceSynchronize();

        unsharpen<<<num_blocks, block_size_1d>>>(image, blurred_unsharpen, image_unsharpen, 0.5, N * N);
        cudaDeviceSynchronize();

        combine<<<num_blocks, block_size_1d>>>(image_unsharpen, blurred_large, mask_large, image2, N * N);
        cudaDeviceSynchronize();

        combine<<<num_blocks, block_size_1d>>>(image2, blurred_small, mask_small, image3, N * N);
        
        // Extra
        // combine<<<num_blocks, block_size_1d>>>(blurred_small, blurred_large, blurred_unsharpen, image3, N * N);

        cudaDeviceSynchronize();

        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        tot += tmp;

        if (debug) {
            std::cout << "  gpu result=[";
            for (int j = 0; j < 10; j++) {
                std::cout << image3[j] << ", ";
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
