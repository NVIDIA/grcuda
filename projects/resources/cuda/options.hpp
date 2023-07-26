// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <getopt.h>

#include <cstdlib>
#include <map>
#include <string>

#include "utils.hpp"

//////////////////////////////
//////////////////////////////

#define CPU_VALIDATION false
#define DEBUG false
#define NUM_ITER 30
#define DEFAULT_BLOCK_SIZE_1D 32
#define DEFAULT_BLOCK_SIZE_2D 8
#define DEFAULT_NUM_BLOCKS 64
#define DEFAULT_SKIP 3
#define DEFAULT_POLICY "async"
#define DEFAULT_PREFETCH false
#define DEFAULT_STREAM_ATTACH false
#define DEFAULT_MAX_DEVICES 1
#define DEFAULT_NVPROF false
// In some benchmarks, allow the computation to be split into an arbitrary number of partitions;
#define DEFAULT_NUM_PARTITIONS 16

//////////////////////////////
//////////////////////////////

enum Policy {
    Sync,
    Async,
    CudaGraph,
    CudaGraphAsync,
    CudaGraphSingle
};

enum BenchmarkEnum {
    B1,
    B5,
    B6,
    B7,
    B8,
    B10,
    B1M,
    B5M,
    B6M,
    B9M,
    B11M,
    B12M,
    B13M,
    ERR
};

//////////////////////////////
//////////////////////////////

inline Policy get_policy(std::string policy) {
    if (policy == "sync")
        return Policy::Sync;
    else if (policy == "cudagraph")
        return Policy::CudaGraph;
    else if (policy == "cudagraphmanual")
        return Policy::CudaGraphAsync;
    else if (policy == "cudagraphsingle")
        return Policy::CudaGraphSingle;
    else
        return Policy::Async;
}

inline BenchmarkEnum get_benchmark(std::string benchmark) {
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
    else if (benchmark == "b1m")
        return BenchmarkEnum::B1M;
    else if (benchmark == "b5m")
        return BenchmarkEnum::B5M;
    else if (benchmark == "b6m")
        return BenchmarkEnum::B6M;
    else if (benchmark == "b9m")
        return BenchmarkEnum::B9M;
    else if (benchmark == "b11m")
        return BenchmarkEnum::B11M;
    else if (benchmark == "b12m")
        return BenchmarkEnum::B12M;
    else if (benchmark == "b13m")
        return BenchmarkEnum::B13M;
    else
        return BenchmarkEnum::ERR;
}

struct Options {
    // Testing options;
    uint num_iter = NUM_ITER;
    int debug = DEBUG;
    int block_size_1d = DEFAULT_BLOCK_SIZE_1D;
    int block_size_2d = DEFAULT_BLOCK_SIZE_2D;
    int num_blocks = DEFAULT_NUM_BLOCKS;
    int N = 0;
    int max_devices = DEFAULT_MAX_DEVICES;
    int skip_iterations = DEFAULT_SKIP;
    bool prefetch = DEFAULT_PREFETCH;
    bool stream_attach = DEFAULT_STREAM_ATTACH;
    bool nvprof = DEFAULT_NVPROF;
    int num_partitions = DEFAULT_NUM_PARTITIONS;
    BenchmarkEnum benchmark_choice = BenchmarkEnum::ERR;
    Policy policy_choice = get_policy(DEFAULT_POLICY);

    // Used for printing;
    std::map<BenchmarkEnum, std::string> benchmark_map;
    std::map<Policy, std::string> policy_map;

    //////////////////////////////
    //////////////////////////////

    Options(int argc, char *argv[]) {
        map_init(policy_map)(Policy::Sync, "sync")(Policy::Async, "async")(Policy::CudaGraph, "cudagraph")(Policy::CudaGraphAsync, "cudagraphmanual")(Policy::CudaGraphSingle, "cudagraphsingle");
        map_init(benchmark_map)
                (BenchmarkEnum::B1, "b1 - VEC")
                (BenchmarkEnum::B5, "b5 - B&S")
                (BenchmarkEnum::B6, "b6 - ML")
                (BenchmarkEnum::B7, "b7 - HITS")
                (BenchmarkEnum::B8, "b8 - IMG")
                (BenchmarkEnum::B10, "b10 - DL")
                (BenchmarkEnum::B1M, "b1m - VEC")
                (BenchmarkEnum::B5M, "b5m - B&S")
                (BenchmarkEnum::B6M, "b6m - ML")
                (BenchmarkEnum::B9M, "b9m - CG")
                (BenchmarkEnum::B11M, "b11m - MUL")
                (BenchmarkEnum::B12M, "b12m - LANCZOS")
                (BenchmarkEnum::B13M, "b13m - MMUL");

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
                                               {"prefetch", no_argument, 0, 'r'},
                                               {"attach", no_argument, 0, 'a'},
                                               {"max_devices", required_argument, 0, 'm'},
                                               {"nvprof", no_argument, 0, 'v'},
                                               {"partitions", required_argument, 0, 'P'},
                                               {0, 0, 0, 0}};
        // getopt_long stores the option index here;
        int option_index = 0;

        while ((opt = getopt_long(argc, argv, "dt:n:b:c:g:s:k:p:ram:vP:", long_options, &option_index)) != EOF) {
            switch (opt) {
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
                case 'r':
                    prefetch = true;
                    break;
                case 'a':
                    stream_attach = true;
                    break;
                case 'm':
                    max_devices = atoi(optarg);
                    break;
                case 'v':
                    nvprof = true;
                    break;
                case 'P':
                    num_partitions = atoi(optarg);
                    break;
                default:
                    break;
            }
        }
    }
};