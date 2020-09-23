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
        std::cout << "running b8 default" << std::endl;
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

    // Initialize arrays;
    start = clock_type::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            image[i * N + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    gaussian_kernel(kernel_small, kernel_small_diameter, 1);
    gaussian_kernel(kernel_large, kernel_large_diameter, 10);
    gaussian_kernel(kernel_unsharpen, kernel_unsharpen_diameter, 5);

    // Create streams;
    cudaStream_t s1, s2, s3, s4, s5;
    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);
    err = cudaStreamCreate(&s3);
    err = cudaStreamCreate(&s4);
    err = cudaStreamCreate(&s5);
    if (err) std::cout << err << std::endl;

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
        
        dim3 block_size_2d_dim(block_size_2d, block_size_2d);
        dim3 grid_size(num_blocks, num_blocks);
        int nb = num_blocks / 2;
        dim3 grid_size_2(nb, nb);

        start = clock_type::now();

        cudaStreamAttachMemAsync(s1, blurred_small, 0);
        cudaStreamAttachMemAsync(s1, mask_small, 0);
        cudaStreamAttachMemAsync(s2, blurred_large, 0);
        cudaStreamAttachMemAsync(s2, mask_large, 0);
        cudaStreamAttachMemAsync(s3, blurred_unsharpen, 0);
        cudaStreamAttachMemAsync(s3, image_unsharpen, 0);
        cudaStreamAttachMemAsync(s2, image2, 0);
        cudaStreamAttachMemAsync(s1, image3, 0);

        gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_small_diameter * kernel_small_diameter * sizeof(float), s1>>>(image, blurred_small, N, N, kernel_small, kernel_small_diameter);

        gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_large_diameter * kernel_large_diameter * sizeof(float), s2>>>(image, blurred_large, N, N, kernel_large, kernel_large_diameter);

        gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_unsharpen_diameter * kernel_unsharpen_diameter * sizeof(float), s3>>>(image, blurred_unsharpen, N, N, kernel_unsharpen, kernel_unsharpen_diameter);

        sobel<<<grid_size_2, block_size_2d_dim, 0, s1>>>(blurred_small, mask_small, N, N);

        sobel<<<grid_size_2, block_size_2d_dim, 0, s2>>>(blurred_large, mask_large, N, N);

        cudaEvent_t e1, e2, e3, e4, e5;
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventCreate(&e3);
        cudaEventCreate(&e4);
        cudaEventCreate(&e5);

        cudaEventRecord(e1, s2);
        cudaStreamWaitEvent(s5, e1, 0);
        maximum_kernel<<<num_blocks, block_size_1d, 0, s5>>>(mask_large, maximum, N * N);

        cudaStreamWaitEvent(s4, e1, 0);
        minimum_kernel<<<num_blocks, block_size_1d, 0, s4>>>(mask_large, minimum, N * N);

        cudaEventRecord(e2, s4);
        cudaEventRecord(e5, s5);

        cudaStreamWaitEvent(s2, e2, 0);
        cudaStreamWaitEvent(s2, e5, 0);

        extend<<<num_blocks, block_size_1d, 0, s2>>>(mask_large, minimum, maximum, N * N);

        unsharpen<<<num_blocks, block_size_1d, 0, s3>>>(image, blurred_unsharpen, image_unsharpen, 0.5, N * N);
        cudaEventRecord(e3, s3);
        cudaStreamWaitEvent(s2, e3, 0);
        combine<<<num_blocks, block_size_1d, 0, s2>>>(image_unsharpen, blurred_large, mask_large, image2, N * N);
        cudaEventRecord(e4, s2);
        cudaStreamWaitEvent(s1, e4, 0);
        cudaStreamAttachMemAsync(s1, image2, 0);

        combine<<<num_blocks, block_size_1d, 0, s1>>>(image2, blurred_small, mask_small, image3, N * N);

        // Extra
        // cudaEventRecord(e1, s2);
        // cudaEventRecord(e2, s3);
        // cudaStreamWaitEvent(s1, e1, 0);
        // cudaStreamWaitEvent(s1, e2, 0);
        // combine<<<num_blocks, block_size_1d, 0, s1>>>(blurred_small, blurred_large, blurred_unsharpen, image3, N * N);

        cudaStreamSynchronize(s1);
        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        if (i >= skip_iterations) tot += tmp;

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
    
    if (debug) std::cout << "\nmean exec time=" << (float) tot / (1000 * (num_executions - skip_iterations)) << " ms" << std::endl;
}
