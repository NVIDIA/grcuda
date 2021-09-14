#pragma once

#include <getopt.h>

#include <cstdlib>
#include <map>
#include <string>

#include "utils.hpp"

//////////////////////////////
//////////////////////////////

#define DEBUG false
#define DEFAULT_BLOCK_SIZE_1D 32
#define DEFAULT_BLOCK_SIZE_2D 8
#define DEFAULT_NUM_BLOCKS 2
#define DEFAULT_POLICY "default"
#define DEFAULT_PREFETCH false
#define DEFAULT_STREAM_ATTACH false
#define DEFAULT_BLACK_AND_WHITE false
#define DEFAULT_RESIZED_IMAGE_WIDTH 1024
#define RESIZED_IMAGE_WIDTH_OUT_SMALL 40
#define RESIZED_IMAGE_WIDTH_OUT_LARGE DEFAULT_RESIZED_IMAGE_WIDTH

// Input and output folders for images;
#define INPUT_IMAGE_FOLDER "img_in"
#define OUTPUT_IMAGE_FOLDER "img_out"

//////////////////////////////
//////////////////////////////

enum Policy {
    Sync,
    Async,
};

//////////////////////////////
//////////////////////////////

inline Policy get_policy(std::string policy) {
    if (policy == "sync")
        return Policy::Sync;
    else
        return Policy::Async;
}

struct Options {
    // Testing options;
    int debug = DEBUG;
    int block_size_1d = DEFAULT_BLOCK_SIZE_1D;
    int block_size_2d = DEFAULT_BLOCK_SIZE_2D;
    int num_blocks = DEFAULT_NUM_BLOCKS;
    bool prefetch = DEFAULT_PREFETCH;
    bool stream_attach = DEFAULT_STREAM_ATTACH;
    Policy policy_choice = get_policy(DEFAULT_POLICY);

    // Input image for the benchmark;
    std::string input_image;
    // Use black and white processing instead of color processing;
    bool black_and_white = DEFAULT_BLACK_AND_WHITE;
    // Resize input image to this size;
    int resized_image_width = DEFAULT_RESIZED_IMAGE_WIDTH;

    // Optional full input/output paths;
    std::string full_input_path;
    std::string full_output_path_small;
    std::string full_output_path_large;

    // Used for printing;
    std::map<Policy, std::string> policy_map;

    //////////////////////////////
    //////////////////////////////

    Options(int argc, char *argv[]) {
        map_init(policy_map)(Policy::Sync, "sync")(Policy::Async, "default");

        int opt;
        static struct option long_options[] = {{"debug", no_argument, 0, 'd'},
                                               {"block_size_1d", required_argument, 0, 'b'},
                                               {"block_size_2d", required_argument, 0, 'c'},
                                               {"num_blocks", required_argument, 0, 'g'},
                                               {"policy", required_argument, 0, 'p'},
                                               {"prefetch", no_argument, 0, 'r'},
                                               {"attach", no_argument, 0, 'a'},
                                               {"input", required_argument, 0, 'i'},
                                               {"bw", no_argument, 0, 'w'},
                                               {"resized_image_width", required_argument, 0, 'n'},
                                               {"full_input_path", required_argument, 0, 'f'},
                                               {"full_output_path_small", required_argument, 0, 's'},
                                               {"full_output_path_large", required_argument, 0, 'l'},
                                               {0, 0, 0, 0}};
        // getopt_long stores the option index here;
        int option_index = 0;

        while ((opt = getopt_long(argc, argv, "db:c:g:p:rai:wn:f:s:l:", long_options, &option_index)) != EOF) {
            switch (opt) {
                case 'd':
                    debug = true;
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
                case 'p':
                    policy_choice = get_policy(optarg);
                    break;
                case 'r':
                    prefetch = true;
                    break;
                case 'a':
                    stream_attach = true;
                    break;
                case 'i':
                    input_image = optarg;
                    break;
                case 'w':
                    black_and_white = true;
                    break;
                case 'n':
                    resized_image_width = atoi(optarg);
                    break;
                case 'f':
                    full_input_path = optarg;
                    break;
                case 's':
                    full_output_path_small = optarg;
                    break;
                case 'l':
                    full_output_path_large = optarg;
                    break;
                default:
                    break;
            }
        }
    }
};
