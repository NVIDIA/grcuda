#pragma once
#include <set>

#include "benchmark.cuh"

typedef struct callBackData {
    float *n1;
    float *n2;
    int *r1;
    int *r2;
} callBackData_t;

class Benchmark7 : public Benchmark {
   public:
    Benchmark7(Options &options) : Benchmark(options) {}
    void alloc();
    void init();
    void reset();
    void execute_sync(int iter);
    void execute_async(int iter);
    void execute_cudagraph(int iter);
    void execute_cudagraph_manual(int iter);
    std::string print_result(bool short_form = false);

   private:
    int degree = 3;
    int iterations = 5;
    int nnz;

    int *ptr, *idx, *val, *ptr2, *idx2, *val2, *rowCounter1, *rowCounter2, *x, *y, *v;
    int *ptr_tmp, *idx_tmp, *val_tmp, *ptr2_tmp, *idx2_tmp, *val2_tmp;
    float *auth1, *auth2, *hub1, *hub2, *auth_norm, *hub_norm;

    cudaStream_t s1, s2;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    inline void random_coo(int *x, int *y, int *val, int N, int degree) {
        for (int i = 0; i < N; i++) {
            std::set<int> edges;
            while (edges.size() < degree) {
                edges.insert(rand() % N);
            }
            int j = 0;
            for (auto iter = edges.begin(); iter != edges.end(); iter++, j++) {
                x[i * degree + j] = i;
                y[i * degree + j] = *iter;
                val[i * degree + j] = 1;
            }
        }
    }
};
