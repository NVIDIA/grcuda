// Copyright (c) 2021, NECSTLab, Politecnico di Milano. All rights reserved.

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
#include <chrono>
#include <iostream>
#include <string>
#include <cuda_runtime.h> 
#include <math.h>
#include "options.hpp"
#include "utils.hpp"

#define CDEPTH 256

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

class ImagePipeline {
   public:
    ImagePipeline(Options &options) : debug(options.debug),
                                      black_and_white(options.black_and_white),
                                      image_width(options.resized_image_width),
                                      do_prefetch(options.prefetch),
                                      stream_attach(options.stream_attach),
                                      policy(options.policy_choice) {
        if (debug) {
            std::cout << "------------------------------" << std::endl;
            std::cout << "- policy=" << options.policy_map[policy] << std::endl;
            std::cout << "- block size 1d=" << options.block_size_1d << std::endl;
            std::cout << "- block size 2d=" << options.block_size_2d << std::endl;
            std::cout << "- num blocks=" << options.num_blocks << std::endl;
            std::cout << "------------------------------" << std::endl;
        }
        grid_size_2d = dim3(options.num_blocks, options.num_blocks);
        grid_size_1d = dim3(options.num_blocks * 2);
        block_size_2d = dim3(options.block_size_2d, options.block_size_2d);
        block_size_1d = dim3(options.block_size_1d);
    }
    std::string print_result(bool short_form = false);

    // Main execution functions;
    void run(unsigned char* input_image);

   private:

    // Instance-specific settings;
    bool black_and_white = DEFAULT_BLACK_AND_WHITE;  // Convert image to black and white;
    int image_width = DEFAULT_RESIZED_IMAGE_WIDTH;
    
    // General configuration settings;
    int debug = DEBUG;
    bool do_prefetch = DEFAULT_PREFETCH;
    bool stream_attach = DEFAULT_STREAM_ATTACH;
    int pascalGpu = 1;
    Policy policy;
    int err = 0;
    dim3 grid_size_2d;
    dim3 grid_size_1d;
    dim3 block_size_2d;
    dim3 block_size_1d;

    // Computation-specific settings;
    int kernel_small_diameter = 7;
    int kernel_large_diameter = 9;
    int kernel_unsharpen_diameter = 7;
    float kernel_small_variance = 0.1;
    float kernel_large_variance = 20;
    float kernel_unsharpen_variance = 5;
    float unsharpen_amount = 30;

    // GPU data;
    int *image, *image3;
    float *image2, *image_unsharpen, *mask_small, *mask_large, *blurred_small, *blurred_large, *blurred_unsharpen;
    float *kernel_small, *kernel_large, *kernel_unsharpen, *maximum_1, *minimum_1, *maximum_2, *minimum_2;
    int *lut[3];
    cudaStream_t s1, s2, s3, s4, s5;

    // Utility functions;
    void alloc();
    void init(unsigned char* input_image, int channel);
    void execute_sync(int channel);
    void execute_async(int channel);
    void run_inner(unsigned char* input_image, int channel);

    inline void gaussian_kernel(float *kernel, int diameter, float sigma) {
        int mean = diameter / 2;
        float sum_tmp = 0;
        for (int i = 0; i < diameter; i++) {
            for (int j = 0; j < diameter; j++) {
                kernel[i * diameter + j] = exp(-0.5 * ((i - mean) * (i - mean) + (j - mean) * (j - mean)) / (sigma * sigma));
                sum_tmp += kernel[i * diameter + j];
            }
        }
        for (int i = 0; i < diameter; i++) {
            for (int j = 0; j < diameter; j++) {
                kernel[i * diameter + j] /= sum_tmp;
            }
        }
    }

    // Beziér curve defined by 3 points.
    // The input is used to map points of the curve to the output LUT,
    // and can be used to combine multiple LUTs.
    // By default, it is just [0, 1, ..., 255];
    inline void spline3(int *input, int *lut, float P[3]) {
        for (int i = 0; i < CDEPTH; i++) {
            float t = float(i) / CDEPTH;
            float x = powf((1 - t), 2) * P[0] + 2 * t * (1 - t) * P[1] + powf(t, 2) * P[2];
            lut[i] = input[int(x * CDEPTH)];
        }
    }

    // Beziér curve defined by 5 points;
    inline void spline5(int *input, int *lut, float P[5]) {
        for (int i = 0; i < CDEPTH; i++) {
            float t = float(i) / CDEPTH;
            float x = powf((1 - t), 4) * P[0] + 4 * t * powf((1 - t), 3) * P[1] + 6 * powf(t, 2) * powf((1 - t), 2) * P[2] + 4 * powf(t, 3) * (1 - t) * P[3] + powf(t, 4) * P[4];
            lut[i] = input[int(x * CDEPTH)];
        }
    }
    
    inline void lut_r(int* lut) {
        // Create a temporary LUT to swap values;
        int *lut_tmp = (int*) malloc(sizeof(int) * CDEPTH);
        // Initialize LUT;
        for (int i = 0; i < CDEPTH; i++) {
            lut[i] = i;
        }
        // Apply 1st curve;
        float P[3] = {0.0, 0.2, 1.0};
        spline3(lut, lut_tmp, P);
        // Apply 2nd curve;
        float P2[5] = {0.0, 0.3, 0.5, 0.99, 1};
        spline5(lut_tmp, lut, P2);
        free(lut_tmp);        
    }

    inline void lut_g(int* lut) {
        // Create a temporary LUT to swap values;
        int *lut_tmp = (int*) malloc(sizeof(int) * CDEPTH);
        // Initialize LUT;
        for (int i = 0; i < CDEPTH; i++) {
            lut[i] = i;
        }
        // Apply 1st curve;
        float P[5] = {0.0, 0.01, 0.5, 0.99, 1};
        spline5(lut, lut_tmp, P);
        // Apply 2nd curve;
        float P2[5] = {0.0, 0.1, 0.5, 0.75, 1};
        spline5(lut_tmp, lut, P2);
        free(lut_tmp);
    }

    inline void lut_b(int* lut) {
        // Create a temporary LUT to swap values;
        int *lut_tmp = (int*) malloc(sizeof(int) * CDEPTH);
        // Initialize LUT;
        for (int i = 0; i < CDEPTH; i++) {
            lut[i] = i;
        }
        // Apply 1st curve;
        float P[5] = {0.0, 0.01, 0.5, 0.99, 1};
        spline5(lut, lut_tmp, P);
        // Apply 2nd curve;
        float P2[5] = {0.0, 0.25, 0.5, 0.70, 1};
        spline5(lut_tmp, lut, P2);
        free(lut_tmp);
    }

// Outdated LUTs;
// #define FACTOR 0.8
//     inline void lut_r(int* lut) {
//         for (int i = 0; i < CDEPTH; i++) {
//             float x = float(i) / CDEPTH;
//             float y = x;
//             // if (i < CDEPTH / 2) {
//                 // y = 0.8 * (1 / (1 + expf(-x + 0.5) * 7 * FACTOR)) + 0.2;
//             // } else {
//                 y = 1 / (1 + expf((-x + 0.5) * 7 * FACTOR));
//             // }
//             lut[i] = std::min(CDEPTH - 1, int(255 * y));
//         }
//     }

//     inline void lut_g(int* lut) {
//         for (int i = 0; i < CDEPTH; i++) {
//             float x = float(i) / CDEPTH;
//             float y = x;
//             // if (i < CDEPTH / 2) {
//                 // y = 0.7 * (1 / (1 + expf(-x + 0.5) * 10 * FACTOR)) + 0.3;
//             // } else {
//                 y = 1 / (1 + expf((-x + 0.5) * 10 * FACTOR));
//             // }
//             lut[i] = std::min(CDEPTH - 1, int(255 * powf(y, 1.6)));
//         }
//     }

//     inline void lut_b(int* lut) {
//         for (int i = 0; i < CDEPTH; i++) {
//             float x = float(i) / CDEPTH;
//             float y = x;
//             // if (i < CDEPTH / 2) {
//             //     y = 0.8 * (1 / (1 + expf(-x + 0.5) * 10 * FACTOR)) + 0.2;
//             // } else {
//                 y = 1 / (1 + expf((-x + 0.5) * 9 * FACTOR));
//             // }
//             lut[i] = std::min(CDEPTH - 1, int(255 * powf(y, 1.4)));
//         }
//     }

// img_out = img.copy()
// lut_b = lambda x: 0.7 * (1 / (1 + np.exp((-x + 0.5) * 10))) + 0.3 if x < 0.5 else 1 / (1 + np.exp((-x + 0.5) * 10))
// lut_r = lambda x: 0.8 * (1 / (1 + np.exp((-x + 0.5) * 7))) + 0.2 if x < 0.5 else (1 / (1 + np.exp((-x + 0.5) * 7)))
// lut_g = lambda x: 0.8 * (1 / (1 + np.exp((-x + 0.5) * 10))) + 0.2 if x < 0.5 else  (1 / (1 + np.exp((-x + 0.5) * 9)))
// lut_g2 = lambda x: x**1.4
// lut_b2 = lambda x: x**1.6
// img_out[:, :, 0] = np.vectorize(lut_b)(img[:, :, 0])
// img_out[:, :, 1] = np.vectorize(lut_g)(img[:, :, 1])
// img_out[:, :, 2] = np.vectorize(lut_r)(img[:, :, 2])

// img_out[:, :, 1] = np.vectorize(lut_g2)(img_out[:, :, 1])
// img_out[:, :, 0] = np.vectorize(lut_b2)(img_out[:, :, 0])

};
