#pragma once
#include "benchmark.cuh"

class Benchmark6 : public Benchmark {
   public:
    Benchmark6(Options &options) : Benchmark(options) {}
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
    int num_features = 200;
    int num_classes = 10;
    int *x;
    float *z;
    float *nb_feat_log_prob, *nb_class_log_prior, *ridge_coeff, *ridge_intercept, *nb_amax, *nb_l, *r1, *r2;
    int *r;
    cudaStream_t s1, s2;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7, kernel_8, kernel_9, kernel_10;
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
};
