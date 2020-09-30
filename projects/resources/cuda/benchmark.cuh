#pragma once
#include <chrono>
#include <iostream>
#include <string>

#include "options.hpp"
#include "utils.hpp"

struct Benchmark {
   public:
    virtual void alloc() = 0;
    virtual void init() = 0;
    virtual void reset() = 0;
    virtual void execute_async(int iter) = 0;
    virtual void execute_sync(int iter) = 0;
    virtual void execute_cudagraph(int iter) = 0;
    virtual void execute_cudagraph_manual(int iter) = 0;
    virtual void execute_cudagraph_single(int iter) = 0;
    virtual std::string print_result(bool short_form = false) = 0;
    void run();
    int add_node(void **paramarray, cudaKernelNodeParams &param, void *func, dim3 gridsize, dim3 threads, cudaGraph_t &g, cudaGraphNode_t *n, std::vector<cudaGraphNode_t> &dependencies, int shared_memory = 0);

    Benchmark(Options &options) : debug(options.debug),
                                  num_executions(options.num_iter),
                                  N(options.N),
                                  block_size_1d(options.block_size_1d),
                                  block_size_2d(options.block_size_2d),
                                  num_blocks(options.num_blocks),
                                  skip_iterations(options.skip_iterations),
                                  do_prefetch(options.prefetch),
                                  stream_attach(options.stream_attach),
                                  policy(options.policy_choice),
                                  benchmark_name(options.benchmark_choice) {
        cudaDeviceGetAttribute(&pascalGpu, cudaDeviceAttr::cudaDevAttrConcurrentManagedAccess, 0);
        if (debug) {
            std::cout << "------------------------------" << std::endl;
            std::cout << "- running " << options.benchmark_map[benchmark_name] << std::endl;
            std::cout << "- num executions=" << num_executions << std::endl;
            std::cout << "- iterations to skip=" << skip_iterations << std::endl;
            std::cout << "- N=" << N << std::endl;
            std::cout << "- policy=" << options.policy_map[policy] << std::endl;
            std::cout << "- block size 1d=" << block_size_1d << std::endl;
            std::cout << "- block size 2d=" << block_size_2d << std::endl;
            std::cout << "- num blocks=" << num_blocks << std::endl;
            std::cout << "------------------------------" << std::endl;
        }
    }

    virtual ~Benchmark(){};

   protected:
    int debug = DEBUG;
    int num_executions = NUM_ITER;
    int N = 0;
    int block_size_1d = DEFAULT_BLOCK_SIZE_1D;
    int block_size_2d = DEFAULT_BLOCK_SIZE_2D;
    int num_blocks = DEFAULT_NUM_BLOCKS;
    int skip_iterations = 0;
    bool do_prefetch = DEFAULT_PREFETCH;
    bool stream_attach = DEFAULT_STREAM_ATTACH;
    int pascalGpu = 0;
    Policy policy;
    BenchmarkEnum benchmark_name;
    int err = 0;
};
