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
    std::string print_result(bool short_form = false);

   private:
    int num_features = 200;
    int num_classes = 10;
    int *x;
    float *z;
    float *nb_feat_log_prob, *nb_class_log_prior, *ridge_coeff, *ridge_intercept, *nb_amax, *nb_l, *r1, *r2;
    int *r;
    cudaStream_t s1, s2;
};
