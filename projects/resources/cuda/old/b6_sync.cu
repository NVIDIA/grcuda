#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include "utils.hpp"
#include "options.hpp"
#include "b6.cuh"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

void reset(float *r1, float *r2, const float *nb_class_log_prior, int N, int num_classes) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < num_classes; j++) {
            r1[i * num_classes + j] = nb_class_log_prior[j];
            r2[i * num_classes + j] = 0;
        }
        // r1_mean[i] = 0;
        // r1_std[i] = 0;
    }
}

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

    //srand(time(0));
    srand(13);

    Options options = Options(argc, argv);
	int debug = options.debug;

    int num_executions = options.num_iter;
    int N = options.N;

    int block_size = options.block_size_1d;
    int num_blocks = 32; // options.num_blocks;
    int skip_iterations = options.skip_iterations;
    int err = 0;

    int num_features = 200;
    int num_classes = 10;

    if (debug) {
        std::cout << "running b6 sync" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();
    int *x;
    float *z;
    err = cudaMallocManaged(&x, sizeof(int) * N * num_features);
    if (err) std::cout << err << std::endl;
    err = cudaMallocManaged(&z, sizeof(float) * N * num_features);
    if (err) std::cout << err << std::endl;
    
    float *nb_feat_log_prob, *nb_class_log_prior, *ridge_coeff, *ridge_intercept, *nb_amax, *nb_l, *r1, *r2;
    // float *r1_mean, *r1_std;
    int *r;
    err = cudaMallocManaged(&nb_feat_log_prob, sizeof(float) * num_classes * num_features);
    err = cudaMallocManaged(&nb_class_log_prior, sizeof(float) * num_classes);
    err = cudaMallocManaged(&ridge_coeff, sizeof(float) * num_classes * num_features);
    err = cudaMallocManaged(&ridge_intercept, sizeof(float) * num_classes);
    err = cudaMallocManaged(&nb_amax, sizeof(float) * N);
    err = cudaMallocManaged(&nb_l, sizeof(float) * N);
    err = cudaMallocManaged(&r1, sizeof(float) * N * num_classes);
    err = cudaMallocManaged(&r2, sizeof(float) * N * num_classes);
    err = cudaMallocManaged(&r, sizeof(int) * N);

    // err = cudaMallocManaged(&r1_mean, sizeof(float) * num_features);
    // err = cudaMallocManaged(&r1_std, sizeof(float) * num_features);
    if (err) std::cout << err << std::endl;

    // Initialze arrays;
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < num_features; j++) {
            nb_feat_log_prob[i * num_features + j] = (float)(rand()) / (float)(RAND_MAX);
            ridge_coeff[i * num_features + j] = (float)(rand()) / (float)(RAND_MAX);
        }
        nb_class_log_prior[i] = (float)(rand()) / (float)(RAND_MAX);
        ridge_intercept[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    int max_occurrence_of_ngram = 2;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < num_features; j++) {
            x[i * num_features + j] = rand() % max_occurrence_of_ngram;
        }
        for (int j = 0; j < num_classes; j++) {
            r1[i * num_classes + j] = nb_class_log_prior[j];
            r2[i * num_classes + j] = 0;
        }
    }

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset(r1, r2, nb_class_log_prior, N, num_classes);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << " reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();
        rr_1<<<num_blocks, block_size>>>(x, z, N, num_features);
        // dim3 num_blocks_2d(8, 8);
        // dim3 block_size_2d(1, 32);
        // rr_1_0<<<num_blocks_2d, block_size_2d>>>(x, r1_mean, r1_std, N, num_features);
        // cudaDeviceSynchronize();
        // rr_1_1<<<num_blocks_2d, block_size_2d>>>(x, z, r1_mean, r1_std, N, num_features);
        cudaDeviceSynchronize();

        // auto e1 = clock_type::now();
        // auto rr1time = chrono::duration_cast<chrono::microseconds>(e1 - start).count();
        // if (debug) std::cout << " rr1=" << (float) rr1time / 1000 << " ms" << std::endl;
       
        nb_1<<<num_blocks, block_size>>>(x, nb_feat_log_prob, r1, N, num_features, num_classes);
        cudaDeviceSynchronize();
       
        rr_2<<<num_blocks, block_size>>>(z, ridge_coeff, r2, N, num_features, num_classes);
        cudaDeviceSynchronize();
       
        nb_2<<<num_blocks, block_size>>>(r1, nb_amax, N, num_classes);
        cudaDeviceSynchronize();
       
        nb_3<<<num_blocks, block_size>>>(r1, nb_amax, nb_l, N, num_classes);
        cudaDeviceSynchronize();
       
        rr_3<<<num_blocks, block_size>>>(r2, ridge_intercept, N, num_classes);
        cudaDeviceSynchronize();
       
        nb_4<<<num_blocks, block_size>>>(r1, nb_l, N, num_classes);
        cudaDeviceSynchronize();
       
        softmax<<<num_blocks, block_size>>>(r1, N, num_classes);
        cudaDeviceSynchronize();
        
        softmax<<<num_blocks, block_size>>>(r2, N, num_classes);
        cudaDeviceSynchronize();
       
        argmax<<<num_blocks, block_size>>>(r1, r2, r, N, num_classes);
        cudaDeviceSynchronize();
       
        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        if (i >= skip_iterations) tot += tmp;

        if (debug) {
            std::cout << "  gpu result=[";
            for (int j = 0; j < 10; j++) {
                std::cout << r[j] << ", ";
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
