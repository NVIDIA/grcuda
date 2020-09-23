#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include <set>
#include "utils.hpp"
#include "options.hpp"
#include "b10.cuh"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

    srand(time(0));

    Options options = Options(argc, argv);
	int debug = options.debug;

    int num_executions = options.num_iter;
    int N = options.N;

    int K = 3;
    int channels = 1;
    int stride = 2;
    int kn1 = 8;
    int kn2 = 16;
    int pooling_diameter = 5;

    int block_size_1d = options.block_size_1d;
    int block_size_2d = options.block_size_2d;
    int num_blocks = options.num_blocks;
    int skip_iterations = options.skip_iterations;
    int err = 0;
    if (debug) {
        std::cout << "running b10 sync" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size_1d << std::endl;
        std::cout << "block size 2d=" << block_size_2d << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();
    float *x, *x1, *x2, *x3, *y, *y1, *y2, *y3, *kernel_1, *kernel_2, *kernel_3, *kernel_4, *z, *dense_weights, *res;
    float *x11, *y11;
    float *x_cpu = (float *) malloc(sizeof(float) * N * N * channels);
    float *y_cpu = (float *) malloc(sizeof(float) * N * N * channels);
    int x_len = N * N * channels;
    int x1_len = (N / stride) * (N / stride) * kn1;
    int pooled_len = x1_len / (pooling_diameter * pooling_diameter);
    int x2_len = ((N / stride) / pooling_diameter / stride) * ((N / stride) / pooling_diameter / stride) * kn2;
    // int x2_len = (N / (stride * stride)) * (N / (stride * stride)) * kn2;
    int x3_len = kn2;
    err = cudaMallocManaged(&x, sizeof(float) * x_len);
    err = cudaMallocManaged(&x1, sizeof(float) * x1_len);
    err = cudaMallocManaged(&x2, sizeof(float) * x2_len);
    err = cudaMallocManaged(&x3, sizeof(float) * x3_len);

    err = cudaMallocManaged(&y, sizeof(float) * x_len);
    err = cudaMallocManaged(&y1, sizeof(float) * x1_len);
    err = cudaMallocManaged(&y2, sizeof(float) * x2_len);
    err = cudaMallocManaged(&y3, sizeof(float) * x3_len);

    int k1_len = channels * K * K * kn1;
    int k2_len = kn1 * K * K * kn2;
    err = cudaMallocManaged(&kernel_1, sizeof(float) * k1_len);
    err = cudaMallocManaged(&kernel_2, sizeof(float) * k2_len);
    err = cudaMallocManaged(&kernel_3, sizeof(float) * k1_len);
    err = cudaMallocManaged(&kernel_4, sizeof(float) * k2_len);

    int z_len = 2 * x2_len;
    err = cudaMallocManaged(&z, sizeof(float) * z_len);
    err = cudaMallocManaged(&dense_weights, sizeof(float) * z_len);
    err = cudaMallocManaged(&res, sizeof(float));   

    
    err = cudaMallocManaged(&x11, sizeof(float) * pooled_len);
    err = cudaMallocManaged(&y11, sizeof(float) * pooled_len);
   
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;
    start = clock_type::now();
    for (int i = 0; i < x_len; i++) {
        x_cpu[i] = (float)(rand()) / (float)(RAND_MAX);
        y_cpu[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    for (int i = 0; i < k1_len; i++) {
        kernel_1[i] = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
        kernel_3[i] = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
    }
    for (int i = 0; i < k2_len; i++) {
        kernel_2[i] = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
        kernel_4[i] = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
    }

    for (int i = 0; i < z_len; i++) {
        dense_weights[i] = (((float)(rand()) / (float)(RAND_MAX)) * 2 - 1) / z_len;
    }

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset(x, y, x_cpu, y_cpu, x_len, res);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();

        dim3 block_size_2d_dim(block_size_2d, block_size_2d);
        dim3 grid_size(num_blocks, num_blocks);
        dim3 grid_size_2(num_blocks / 2, num_blocks / 2);

        dim3 block_size_3d_dim(block_size_2d / 2, block_size_2d / 2, block_size_2d / 2);
        dim3 grid_size_3(num_blocks / 2, num_blocks / 2, num_blocks / 2);

        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float)>>>(x1, x, kernel_1, N, N, channels, K, kn1, stride);
        cudaDeviceSynchronize();
        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float)>>>(y1, y, kernel_3, N, N, channels, K, kn1, stride);
        cudaDeviceSynchronize();

        mean_pooling<<<grid_size_3, block_size_3d_dim>>>(x11, x1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);
        cudaDeviceSynchronize();
        mean_pooling<<<grid_size_3, block_size_3d_dim>>>(y11, y1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);
        cudaDeviceSynchronize();

        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float)>>>(x2, x11, kernel_2, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);
        cudaDeviceSynchronize();
        conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float)>>>(y2, y11, kernel_4, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);
        cudaDeviceSynchronize();

        // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float)>>>(x2, x1, kernel_2, N / stride, N / stride, kn1, K, kn2, stride);
        // cudaDeviceSynchronize();
        // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float)>>>(y2, y1, kernel_4, N / stride, N / stride, kn1, K, kn2, stride);
        // cudaDeviceSynchronize();

        // gap<<<grid_size_2, block_size_2d_dim, kn2 * sizeof(float)>>>(x3, x2, N / (stride * stride), N / (stride * stride), kn2);
        // cudaDeviceSynchronize();
        // gap<<<grid_size_2, block_size_2d_dim, kn2 * sizeof(float)>>>(y3, y2, N / (stride * stride), N / (stride * stride), kn2);
        // cudaDeviceSynchronize();

        concat<<<num_blocks, block_size_1d>>>(z, x2, y2, x2_len);
        cudaDeviceSynchronize();
        
        dot_product<<<num_blocks, block_size_1d>>>(z, dense_weights, res, x2_len);
        cudaDeviceSynchronize();

        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        tot += tmp;

        if (debug) {
            std::cout << "  gpu result=" << *res << "; time=" << (float) tmp / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << *res << "," << (float) (reset_time + tmp) / 1e6 << "," << (float) reset_time / 1e6 << "," << (float) tmp / 1e6 << std::endl;
        }
    }

    // Print;
	cudaDeviceSynchronize();
    
    if (debug) std::cout << "\nmean exec time=" << (float) tot / (1000 * num_executions) << " ms" << std::endl;
}
