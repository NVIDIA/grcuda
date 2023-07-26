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
#include <sstream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>

#include "options.hpp"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

class OpenCVInterface {
   public:
    OpenCVInterface(Options &options) : debug(options.debug),
                                        image_name(options.input_image),
                                        full_input_path(options.full_input_path),
                                        full_output_path_small(options.full_output_path_small),
                                        full_output_path_large(options.full_output_path_large),
                                        black_and_white(options.black_and_white),
                                        image_width(options.resized_image_width) {

        // Validate input/output values;
        if ((full_input_path.empty() || full_output_path_small.empty() || full_output_path_large.empty()) && image_name.empty()) {
            if (debug) std::cout << "error: you must specify the name of an image in " << INPUT_IMAGE_FOLDER <<
             " (without extension) or specify the full input and output path of the image" << std::endl;
        }                               

        if (debug) {
            std::cout << "------------------------------" << std::endl;
            if (!options.full_input_path.empty()) {
                std::cout << "- input image path=" << options.full_input_path << std::endl;
            } else {
                std::cout << "- image name=" << options.input_image << std::endl;
            }
            std::cout << "- image size=" << image_width << "x" << image_width << std::endl;
            std::cout << "- black and white? " << (options.black_and_white ? "yes" : "no") << std::endl;
            if (!options.full_output_path_small.empty()) {
                std::cout << "- ouput image path, small=" << options.full_output_path_small << std::endl;
            }
            if (!options.full_output_path_large.empty()) {
                std::cout << "- ouput image path, large=" << options.full_output_path_large << std::endl;
            }
            std::cout << "------------------------------" << std::endl;
        }
    }

    // Main execution functions;
    uchar* read_input();
    void write_output(unsigned char* buffer);
    int image_array_length;

   private:

    // Instance-specific settings;
    std::string image_name;  // Input image for the benchmark;
    std::string full_input_path;  // Optional input/output paths to the image;
    std::string full_output_path_small;
    std::string full_output_path_large;
    bool black_and_white = DEFAULT_BLACK_AND_WHITE;  // Convert image to black and white;
    int image_width = DEFAULT_RESIZED_IMAGE_WIDTH;
    
    // General configuration settings;
    int debug = DEBUG;

    // OpenCV data;
    cv::Mat image_matrix;
    cv::Mat resized_image;
    cv::Mat output_matrix;

    // Utility functions;
    void write_output_inner(std::string kind, int resize_width);
};
