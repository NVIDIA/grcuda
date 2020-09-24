#pragma once
#include "benchmark.cuh"

class Benchmark10 : public Benchmark {
   public:
    Benchmark10(Options &options) : Benchmark(options) {}
    void alloc();
    void init();
    void reset();
    void execute_sync(int iter);
    void execute_async(int iter);
    void execute_cudagraph(int iter);
    void execute_cudagraph_manual(int iter);
    std::string print_result(bool short_form = false);

   private:
    int K = 3;
    int channels = 1;
    int stride = 2;
    int kn1 = 8;
    int kn2 = 16;
    int pooling_diameter = 5;

    float *x, *x1, *x2, *x3, *y, *y1, *y2, *y3, *kernel_1, *kernel_2, *kernel_3, *kernel_4, *z, *dense_weights, *res;
    float *x11, *y11;
    float *x_cpu;
    float *y_cpu;
    int x_len;
    int x1_len;
    int pooled_len;
    int x2_len;
    int x3_len;
    int k1_len, k2_len, z_len;

    cudaStream_t s1, s2;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
};
