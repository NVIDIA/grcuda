#pragma once
#include "benchmark.cuh"

class Benchmark5 : public Benchmark {
   public:
    Benchmark5(Options &options) : Benchmark(options) {
        graphs = std::vector<cudaGraph_t>(M);
        graphExec = std::vector<cudaGraphExec_t>(M);
        kernels = std::vector<cudaGraphNode_t>(M);
        kernel_params = std::vector<cudaKernelNodeParams>(M);
    }
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
    int M = 10;
    double **x, **y, *tmp_x;
    cudaStream_t *s;
    std::vector<cudaGraph_t> graphs;
    std::vector<cudaGraphExec_t> graphExec;

    std::vector<cudaGraphNode_t> nodeDependencies;
    std::vector<cudaGraphNode_t> kernels;
    std::vector<cudaKernelNodeParams> kernel_params;
};
