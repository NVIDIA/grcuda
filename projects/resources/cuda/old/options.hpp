#pragma once

#include <getopt.h>
#include <string>
#include <cstdlib>
#include <map>
#include "utils.hpp"

//////////////////////////////
//////////////////////////////

#define DEBUG false
#define NUM_ITER 30
#define DEFAULT_BLOCK_SIZE_1D 32
#define DEFAULT_BLOCK_SIZE_2D 8
#define DEFAULT_NUM_BLOCKS 64
#define DEFAULT_SKIP 3
#define DEFAULT_BENCHMARK "b1"
#define DEFAULT_POLICY "async"

//////////////////////////////
//////////////////////////////

enum Policy
{
    Sync,
    Async,
    CudaGraph,
    CudaGraphAsync
};

enum BenchmarkEnum
{
    B1,
    B5,
    B6,
    B7,
    B8,
    B10,
    ERR
};

//////////////////////////////
//////////////////////////////

inline Policy get_policy(std::string policy)
{
    if (policy == "sync")
        return Policy::Sync;
    else if (policy == "cudagraph")
        return Policy::CudaGraph;
    else if (policy == "cudagraph_manual")
        return Policy::CudaGraphAsync;
    else
        return Policy::Async;
}

inline BenchmarkEnum get_benchmark(std::string benchmark)
{
    if (benchmark == "b1")
        return BenchmarkEnum::B1;
    else if (benchmark == "b5")
        return BenchmarkEnum::B5;
    else if (benchmark == "b6")
        return BenchmarkEnum::B6;
    else if (benchmark == "b7")
        return BenchmarkEnum::B7;
    else if (benchmark == "b8")
        return BenchmarkEnum::B8;
    else if (benchmark == "b10")
        return BenchmarkEnum::B10;
    else
        return BenchmarkEnum::ERR;
}

struct Options
{

    // Testing options;
    uint num_iter = NUM_ITER;
    int debug = DEBUG;
    int block_size_1d = DEFAULT_BLOCK_SIZE_1D;
    int block_size_2d = DEFAULT_BLOCK_SIZE_2D;
    int num_blocks = DEFAULT_NUM_BLOCKS;
    int N = 0;
    int skip_iterations = DEFAULT_SKIP;
    BenchmarkEnum benchmark_choice = get_benchmark(DEFAULT_BENCHMARK);
    Policy policy_choice = get_policy(DEFAULT_POLICY);

    // Used for printing;
    std::map<BenchmarkEnum, std::string> benchmark_map;
    std::map<Policy, std::string> policy_map;

    //////////////////////////////
    //////////////////////////////

    Options(int argc, char *argv[])
    {
        map_init(policy_map)(Policy::Sync, "sync")(Policy::Async, "async")(Policy::CudaGraph, "cudagraph")(Policy::CudaGraphAsync, "cudagraph_manual");
        map_init(benchmark_map)(BenchmarkEnum::B1, "b1")(BenchmarkEnum::B5, "b5")(BenchmarkEnum::B6, "b6")(BenchmarkEnum::B7, "b7")(BenchmarkEnum::B8, "b8")(BenchmarkEnum::B10, "b10");

        int opt;
        static struct option long_options[] = {{"debug", no_argument, 0, 'd'},
                                               {"num_iter", required_argument, 0, 't'},
                                               {"N", required_argument, 0, 'n'},
                                               {"block_size_1d", required_argument, 0, 'b'},
                                               {"block_size_2d", required_argument, 0, 'c'},
                                               {"num_blocks", required_argument, 0, 'g'},
                                               {"skip_first", required_argument, 0, 's'},
                                               {"benchmark", required_argument, 0, 'k'},
                                               {"policy", required_argument, 0, 'p'},
                                               {0, 0, 0, 0}};
        // getopt_long stores the option index here;
        int option_index = 0;

        while ((opt = getopt_long(argc, argv, "dt:n:b:c:g:s:k:p:", long_options, &option_index)) != EOF)
        {
            switch (opt)
            {
            case 'd':
                debug = true;
                break;
            case 't':
                num_iter = atoi(optarg);
                break;
            case 'n':
                N = atoi(optarg);
                break;
            case 'b':
                block_size_1d = atoi(optarg);
                break;
            case 'c':
                block_size_2d = atoi(optarg);
                break;
            case 'g':
                num_blocks = atoi(optarg);
                break;
            case 's':
                skip_iterations = atoi(optarg);
                break;
            case 'k':
                benchmark_choice = get_benchmark(optarg);
                break;
            case 'p':
                policy_choice = get_policy(optarg);
                break;
            default:
                break;
            }
        }
    }
};
