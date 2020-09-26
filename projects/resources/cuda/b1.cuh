#pragma once
#include "benchmark.cuh"

class Benchmark1 : public Benchmark {
   public:
    Benchmark1(Options &options) : Benchmark(options) {}
    void alloc();
    void init();
    void reset();
    void execute_sync(int iter);
    void execute_async(int iter);
    void execute_cudagraph(int iter);
    void execute_cudagraph_manual(int iter);
    void execute_cudagraph_single(int iter);
    void prefetch(cudaStream_t &s1, cudaStream_t &s2);
    std::string print_result(bool short_form = false);

   private:
    float *x, *y, *x1, *y1, *res;
    cudaStream_t s1, s2;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t kernel_1, kernel_2, kernel_3;
    cudaKernelNodeParams kernel_1_params;
    cudaKernelNodeParams kernel_2_params;
    cudaKernelNodeParams kernel_3_params;
};
