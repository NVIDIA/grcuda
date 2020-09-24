#pragma once
#include "benchmark.cuh"

class Benchmark8 : public Benchmark {
   public:
    Benchmark8(Options &options) : Benchmark(options) {}
    void alloc();
    void init();
    void reset();
    void execute_sync(int iter);
    void execute_async(int iter);
    void execute_cudagraph(int iter);
    void execute_cudagraph_manual(int iter);
    void execute_cudagraph_single(int iter);
    std::string print_result(bool short_form = false);

   private:
    int kernel_small_diameter = 3;
    int kernel_large_diameter = 5;
    int kernel_unsharpen_diameter = 3;

    float *image, *image2, *image3, *image_unsharpen, *mask_small, *mask_large, *mask_unsharpen, *blurred_small, *blurred_large, *blurred_unsharpen;
    float *kernel_small, *kernel_large, *kernel_unsharpen, *maximum, *minimum;
    cudaStream_t s1, s2, s3, s4, s5;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7, kernel_8, kernel_9, kernel_10, kernel_11;
    cudaKernelNodeParams kernel_1_params;
    cudaKernelNodeParams kernel_2_params;
    cudaKernelNodeParams kernel_3_params;
    cudaKernelNodeParams kernel_4_params;
    cudaKernelNodeParams kernel_5_params;
    cudaKernelNodeParams kernel_6_params;
    cudaKernelNodeParams kernel_7_params;
    cudaKernelNodeParams kernel_8_params;
    cudaKernelNodeParams kernel_9_params;
    cudaKernelNodeParams kernel_10_params;
    cudaKernelNodeParams kernel_11_params;

    inline void gaussian_kernel(float *kernel, int diameter, float sigma) {
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
};
