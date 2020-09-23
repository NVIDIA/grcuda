#pragma once
#include "benchmark.cuh"

class Benchmark8 : public Benchmark {
   public:
    Benchmark8(Options &options) : Benchmark(options) {}
    void alloc();
    void init();
    void reset();
    void execute_sync(int iter);
    void execute_async(int iter);
    void execute_cudagraph(int iter);
    void execute_cudagraph_manual(int iter);
    std::string print_result(bool short_form = false);

   private:
    int kernel_small_diameter = 3;
    int kernel_large_diameter = 5;
    int kernel_unsharpen_diameter = 3;

    float *image, *image2, *image3, *image_unsharpen, *mask_small, *mask_large, *mask_unsharpen, *blurred_small, *blurred_large, *blurred_unsharpen;
    float *kernel_small, *kernel_large, *kernel_unsharpen, *maximum, *minimum;
    cudaStream_t s1, s2, s3, s4, s5;

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
};
