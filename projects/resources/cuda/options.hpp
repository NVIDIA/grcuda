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

#define DEBUG false
#define NUM_ITER 30
#define DEFAULT_BLOCK_SIZE_1D 32
#define DEFAULT_BLOCK_SIZE_2D 8
#define DEFAULT_NUM_BLOCKS 64
#define DEFAULT_SKIP 3
#define DEFAULT_BENCHMARK "b1"
#define DEFAULT_POLICY "async"
#define DEFAULT_PREFETCH false
#define DEFAULT_STREAM_ATTACH false

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
    int skip_iterations = DEFAULT_SKIP;
    bool prefetch = DEFAULT_PREFETCH;
    bool stream_attach = DEFAULT_STREAM_ATTACH;
    BenchmarkEnum benchmark_choice = get_benchmark(DEFAULT_BENCHMARK);
    Policy policy_choice = get_policy(DEFAULT_POLICY);

    // Used for printing;
    std::map<BenchmarkEnum, std::string> benchmark_map;
    std::map<Policy, std::string> policy_map;

    //////////////////////////////
    //////////////////////////////

    Options(int argc, char *argv[]) {
        map_init(policy_map)(Policy::Sync, "sync")(Policy::Async, "async")(Policy::CudaGraph, "cudagraph")(Policy::CudaGraphAsync, "cudagraphmanual")(Policy::CudaGraphSingle, "cudagraphsingle");
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
                                               {"prefetch", required_argument, 0, 'r'},
                                               {"attach", required_argument, 0, 'a'},
                                               {0, 0, 0, 0}};
        // getopt_long stores the option index here;
        int option_index = 0;

        while ((opt = getopt_long(argc, argv, "dt:n:b:c:g:s:k:p:ra", long_options, &option_index)) != EOF) {
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
                default:
                    break;
            }
        }
    }
};
