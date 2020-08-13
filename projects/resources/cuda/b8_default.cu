#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include <set>
#include "utils.hpp"
#include "options.hpp"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

extern "C" __global__ void gaussian_blur(const float *image, float *result, int rows, int cols, const float* kernel, int diameter) {
    extern __shared__ float kernel_local[];
    for(int i = threadIdx.x; i < diameter; i += blockDim.x) {
        for(int j = threadIdx.y; j < diameter; j += blockDim.y) {
            kernel_local[i * diameter + j] = kernel[i * diameter + j];
        }
    }
    __syncthreads();

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols; j += blockDim.y * gridDim.y) {
            float sum = 0;
            int radius = diameter / 2;
            for (int x = -radius; x <= radius; ++x) {
                for (int y = -radius; y <= radius; ++y) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && ny >= 0 && nx < rows && ny < cols) {
                        sum += kernel_local[(x + radius) * diameter + (y + radius)] * image[nx * cols + ny];
                    }
                }
            }
            result[i * cols + j] = sum;
        }
    }
}

extern "C" __global__ void sobel(const float *image, float *result, int rows, int cols) {
    // int SOBEL_X[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    // int SOBEL_Y[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    __shared__ int SOBEL_X[9];
    __shared__ int SOBEL_Y[9];
    if (threadIdx.x == 0 && threadIdx.y == 0) {   
        SOBEL_X[0] = -1;
        SOBEL_X[1] = -2;
        SOBEL_X[2] = -1;
        SOBEL_X[3] = 0;
        SOBEL_X[4] = 0;
        SOBEL_X[5] = 0;
        SOBEL_X[6] = 1;
        SOBEL_X[7] = 2;
        SOBEL_X[8] = 1;

        SOBEL_Y[0] = -1;
        SOBEL_Y[1] = 0;
        SOBEL_Y[2] = 1;
        SOBEL_Y[3] = -2;
        SOBEL_Y[4] = 0;
        SOBEL_Y[5] = 2;
        SOBEL_Y[6] = -1;
        SOBEL_Y[7] = 0;
        SOBEL_Y[8] = 1;
    }
    __syncthreads();
    
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols; j += blockDim.y * gridDim.y) {
            float sum_gradient_x = 0.0, sum_gradient_y = 0.0;
            int radius = 1;
            for (int x = -radius; x <= radius; ++x) {
                for (int y = -radius; y <= radius; ++y) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && ny >= 0 && nx < rows && ny < cols) {
                        float neighbour = image[nx * cols + ny];
                        int s = (x + radius) * 3 + y + radius;
                        sum_gradient_x += SOBEL_X[s] * neighbour;
                        sum_gradient_y += SOBEL_Y[s] * neighbour;
                    }
                }
            }
            result[i * cols + j] = sqrt(sum_gradient_x * sum_gradient_x + sum_gradient_y * sum_gradient_y);
        }
    }
}

__device__ float atomicMinf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}

__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    // If val is smaller than current, don't do anything, else update the current value atomically;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    }
    return __int_as_float(old);
}

__inline__ __device__ float warp_reduce_max(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__inline__ __device__ float warp_reduce_min(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

extern "C" __global__ void maximum_kernel(const float *in, float* out, int N) {
    int warp_size = 32;
    float maximum = -1000;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) { 
        maximum = max(maximum, in[i]);
    }
    maximum = warp_reduce_max(maximum); // Obtain the max of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMaxf(out, maximum); // The first thread in the warp updates the output;
}

extern "C" __global__ void minimum_kernel(const float *in, float* out, int N) {
    int warp_size = 32;
    float minimum = 1000;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) { 
        minimum = min(minimum, in[i]);
    }
    minimum = warp_reduce_min(minimum); // Obtain the min of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMinf(out, minimum); // The first thread in the warp updates the output;
}

extern "C" __global__ void extend(float *x, const float *minimum, const float *maximum, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        float res_tmp = 5 * (x[i] - *minimum) / (*maximum - *minimum);
        x[i] = res_tmp > 1 ? 1 : res_tmp;
    }
}

extern "C" __global__ void unsharpen(const float *x, const float *y, float *res, float amount, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        float res_tmp = x[i] * (1 + amount) - y[i] * amount;
        res_tmp = res_tmp > 1 ? 1 : res_tmp;
        res[i] = res_tmp < 0 ? 0 : res_tmp;
    }
}

extern "C" __global__ void combine(const float *x, const float *y, const float *mask, float *res, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        res[i] = x[i] * mask[i] + y[i] * (1 - mask[i]);
    }
}

/////////////////////////////
/////////////////////////////

void reset(float *image, float *maximum, float *minimum, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            image[i * N + j] = 0;
        }
    }
    *maximum = 0;
    *minimum = 0;
}

void gaussian_kernel(float* kernel, int diameter, float sigma) {
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

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

    srand(time(0));

    Options options = Options(argc, argv);
	int debug = options.debug;

    int num_executions = options.num_iter;
    int N = options.N;

    int kernel_small_diameter = 3;
    int kernel_large_diameter = 5;
    int kernel_unsharpen_diameter = 3;

    int block_size_1d = options.block_size_1d;
    int block_size_2d = options.block_size_2d;
    int num_blocks = 16; // options.num_blocks;
    int skip_iterations = options.skip_iterations;
    int err = 0;

    if (debug) {
        std::cout << "running b8 default" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size_1d << std::endl;
        std::cout << "block size 2d=" << block_size_2d << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();
    float *image, *image2, *image3, *image_unsharpen, *mask_small, *mask_large, *mask_unsharpen, *blurred_small, *blurred_large, *blurred_unsharpen;
    float *kernel_small, *kernel_large, *kernel_unsharpen, *maximum, *minimum;
    err = cudaMallocManaged(&image, sizeof(float) * N * N);
    err = cudaMallocManaged(&image2, sizeof(float) * N * N);
    err = cudaMallocManaged(&image3, sizeof(float) * N * N);
    err = cudaMallocManaged(&image_unsharpen, sizeof(float) * N * N);
    err = cudaMallocManaged(&mask_small, sizeof(float) * N * N);
    err = cudaMallocManaged(&mask_large, sizeof(float) * N * N);
    err = cudaMallocManaged(&mask_unsharpen, sizeof(float) * N * N);
    err = cudaMallocManaged(&blurred_small, sizeof(float) * N * N);
    err = cudaMallocManaged(&blurred_large, sizeof(float) * N * N);
    err = cudaMallocManaged(&blurred_unsharpen, sizeof(float) * N * N);

    err = cudaMallocManaged(&kernel_small, sizeof(float) * kernel_small_diameter * kernel_small_diameter);
    err = cudaMallocManaged(&kernel_large, sizeof(float) * kernel_large_diameter * kernel_large_diameter);
    err = cudaMallocManaged(&kernel_unsharpen, sizeof(float) * kernel_unsharpen_diameter * kernel_unsharpen_diameter);
    err = cudaMallocManaged(&maximum, sizeof(float));
    err = cudaMallocManaged(&minimum, sizeof(float));
   
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;
    start = clock_type::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            image[i * N + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    gaussian_kernel(kernel_small, kernel_small_diameter, 1);
    gaussian_kernel(kernel_large, kernel_large_diameter, 10);
    gaussian_kernel(kernel_unsharpen, kernel_unsharpen_diameter, 5);

    // Create streams;
    cudaStream_t s1, s2, s3, s4, s5;
    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);
    err = cudaStreamCreate(&s3);
    err = cudaStreamCreate(&s4);
    err = cudaStreamCreate(&s5);
    if (err) std::cout << err << std::endl;

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (float) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    float tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset(image3, maximum, minimum, N);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (float) reset_time / 1000 << " ms" << std::endl;
        
        dim3 block_size_2d_dim(block_size_2d, block_size_2d);
        dim3 grid_size(num_blocks, num_blocks);
        int nb = num_blocks / 2;
        dim3 grid_size_2(nb, nb);

        start = clock_type::now();

        cudaStreamAttachMemAsync(s1, blurred_small, 0);
        cudaStreamAttachMemAsync(s1, mask_small, 0);
        cudaStreamAttachMemAsync(s2, blurred_large, 0);
        cudaStreamAttachMemAsync(s2, mask_large, 0);
        cudaStreamAttachMemAsync(s3, blurred_unsharpen, 0);
        cudaStreamAttachMemAsync(s3, image_unsharpen, 0);
        cudaStreamAttachMemAsync(s2, image2, 0);
        cudaStreamAttachMemAsync(s3, image3, 0);

        gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_small_diameter * kernel_small_diameter * sizeof(float), s1>>>(image, blurred_small, N, N, kernel_small, kernel_small_diameter);

        gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_large_diameter * kernel_large_diameter * sizeof(float), s2>>>(image, blurred_large, N, N, kernel_large, kernel_large_diameter);

        gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_unsharpen_diameter * kernel_unsharpen_diameter * sizeof(float), s3>>>(image, blurred_unsharpen, N, N, kernel_unsharpen, kernel_unsharpen_diameter);

        sobel<<<grid_size_2, block_size_2d_dim, 0, s1>>>(blurred_small, mask_small, N, N);

        sobel<<<grid_size_2, block_size_2d_dim, 0, s2>>>(blurred_large, mask_large, N, N);

        cudaEvent_t e1, e2, e3, e4, e5;
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventCreate(&e3);
        cudaEventCreate(&e4);
        cudaEventCreate(&e5);

        cudaEventRecord(e1, s2);
        cudaStreamWaitEvent(s5, e1, 0);
        maximum_kernel<<<num_blocks, block_size_1d, 0, s5>>>(mask_large, maximum, N * N);

        cudaStreamWaitEvent(s4, e1, 0);
        minimum_kernel<<<num_blocks, block_size_1d, 0, s4>>>(mask_large, minimum, N * N);

        cudaEventRecord(e2, s4);
        cudaEventRecord(e5, s5);

        cudaStreamWaitEvent(s2, e2, 0);
        cudaStreamWaitEvent(s2, e5, 0);

        extend<<<num_blocks, block_size_1d, 0, s2>>>(mask_large, minimum, maximum, N * N);

        unsharpen<<<num_blocks, block_size_1d, 0, s3>>>(image, blurred_unsharpen, image_unsharpen, 0.5, N * N);
        cudaEventRecord(e3, s3);
        cudaStreamWaitEvent(s2, e3, 0);
        combine<<<num_blocks, block_size_1d, 0, s2>>>(image_unsharpen, blurred_large, mask_large, image2, N * N);
        cudaEventRecord(e4, s2);
        cudaStreamWaitEvent(s1, e4, 0);
        cudaStreamAttachMemAsync(s1, image2, 0);

        combine<<<num_blocks, block_size_1d, 0, s1>>>(image2, blurred_small, mask_small, image3, N * N);

        // Extra
        // cudaEventRecord(e1, s2);
        // cudaEventRecord(e2, s3);
        // cudaStreamWaitEvent(s1, e1, 0);
        // cudaStreamWaitEvent(s1, e2, 0);
        // combine<<<num_blocks, block_size_1d, 0, s1>>>(blurred_small, blurred_large, blurred_unsharpen, image3, N * N);

        cudaStreamSynchronize(s1);
        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        if (i >= skip_iterations) tot += tmp;

        if (debug) {
            std::cout << "  gpu result=[";
            for (int j = 0; j < 10; j++) {
                std::cout << image3[j] << ", ";
            } 
            std::cout << ", ...]; time=" << (float) tmp / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << 0.0 << "," << (float) (reset_time + tmp) / 1e6 << "," << (float) reset_time / 1e6 << "," << (float) tmp / 1e6 << std::endl;
        }
    }

    // Print;
	cudaDeviceSynchronize();
    
    if (debug) std::cout << "\nmean exec time=" << (float) tot / (1000 * (num_executions - skip_iterations)) << " ms" << std::endl;
}
