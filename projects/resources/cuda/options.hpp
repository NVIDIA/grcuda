#pragma once

#include <getopt.h>
#include <string>
#include <cstdlib>

//////////////////////////////
//////////////////////////////

#define DEBUG false
#define NUM_ITER 30
#define DEFAULT_BLOCK_SIZE_1D 32
#define DEFAULT_BLOCK_SIZE_2D 8
#define DEFAULT_NUM_BLOCKS 64 
#define DEFAULT_SKIP 3

//////////////////////////////
//////////////////////////////

struct Options {

    // Testing options;
    uint num_iter = NUM_ITER;
    int debug = DEBUG;
    int block_size_1d = DEFAULT_BLOCK_SIZE_1D;
    int block_size_2d = DEFAULT_BLOCK_SIZE_2D;
    int num_blocks = DEFAULT_NUM_BLOCKS;
    int N = 0;
    int skip_iterations = DEFAULT_SKIP;

    //////////////////////////////
    //////////////////////////////

    Options(int argc, char *argv[]) {
        int opt;
        static struct option long_options[] = {{"debug", no_argument, 0, 'd'},
                                               {"num_iter", required_argument, 0, 't'},
                                               {"N", required_argument, 0, 'n'},
                                               {"block_size_1d", required_argument, 0, 'b'},
                                               {"block_size_2d", required_argument, 0, 'c'},
                                               {"num_blocks", required_argument, 0, 'g'},
                                               {"skip_first", required_argument, 0, 's'},
                                               {0, 0, 0, 0}};
        // getopt_long stores the option index here;
        int option_index = 0;

        while ((opt = getopt_long(argc, argv, "dt:n:b:c:g:s:", long_options, &option_index)) != EOF) {
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
                default:
                    break;
            }
        }
    }
};
