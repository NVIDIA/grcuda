#include "image_pipeline.cuh"

//////////////////////////////
// GPU kernels ///////////////
//////////////////////////////

extern "C" __global__ void gaussian_blur(const int *image, float *result, int rows, int cols, const float* kernel, int diameter) {
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
                        sum += kernel_local[(x + radius) * diameter + (y + radius)] * (float(image[nx * cols + ny]) / 255);
                    }
                }
            }
            result[i * cols + j] = sum;
        }
    }
}

extern "C" __global__ void sobel(float *image, float *result, int rows, int cols) {
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

extern "C" __global__ void maximum_kernel(float *in, float* out, int N) {
    int warp_size = 32;
    float maximum = -1000;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) { 
        maximum = max(maximum, in[i]);
    }
    maximum = warp_reduce_max(maximum); // Obtain the max of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMaxf(out, maximum); // The first thread in the warp updates the output;
}

extern "C" __global__ void minimum_kernel(float *in, float* out, int N) {
    int warp_size = 32;
    float minimum = 1000;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) { 
        minimum = min(minimum, in[i]);
    }
    minimum = warp_reduce_min(minimum); // Obtain the min of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMinf(out, minimum); // The first thread in the warp updates the output;
}

extern "C" __global__ void extend(float *x, const float *minimum, const float *maximum, int n, int extend_factor) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        float res_tmp = extend_factor * (x[i] - *minimum) / (*maximum - *minimum);
        x[i] = res_tmp > 1 ? 1 : res_tmp;
    }
}

extern "C" __global__ void unsharpen(const int *x, const float *y, float *res, float amount, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        float res_tmp = (float(x[i]) / 255) * (1 + amount) - y[i] * amount;
        res_tmp = res_tmp > 1 ? 1 : res_tmp;
        res[i] = res_tmp < 0 ? 0 : res_tmp;
    }
}

extern "C" __global__ void combine(const float *x, const float *y, const float *mask, float *res, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        res[i] = x[i] * mask[i] + y[i] * (1 - mask[i]);
    }
}

extern "C" __global__ void combine_lut(const float *x, const float *y, const float *mask, int *res, int n, int* lut) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        int res_tmp = min(CDEPTH - 1, int(CDEPTH * (x[i] * mask[i] + y[i] * (1 - mask[i]))));
        res[i] = lut[res_tmp];
    }
}

//////////////////////////////
// GPU functions /////////////
//////////////////////////////

void ImagePipeline::alloc() {
    err = cudaMallocManaged(&image, sizeof(int) * image_width * image_width);
    err = cudaMallocManaged(&image2, sizeof(float) * image_width * image_width);
    err = cudaMallocManaged(&image3, sizeof(int) * image_width * image_width);
    err = cudaMallocManaged(&image_unsharpen, sizeof(float) * image_width * image_width);
    err = cudaMallocManaged(&mask_small, sizeof(float) * image_width * image_width);
    err = cudaMallocManaged(&mask_large, sizeof(float) * image_width * image_width);
    err = cudaMallocManaged(&blurred_small, sizeof(float) * image_width * image_width);
    err = cudaMallocManaged(&blurred_large, sizeof(float) * image_width * image_width);
    err = cudaMallocManaged(&blurred_unsharpen, sizeof(float) * image_width * image_width);

    err = cudaMallocManaged(&kernel_small, sizeof(float) * kernel_small_diameter * kernel_small_diameter);
    err = cudaMallocManaged(&kernel_large, sizeof(float) * kernel_large_diameter * kernel_large_diameter);
    err = cudaMallocManaged(&kernel_unsharpen, sizeof(float) * kernel_unsharpen_diameter * kernel_unsharpen_diameter);

    err = cudaMallocManaged(&lut[0], sizeof(int) * CDEPTH);
    err = cudaMallocManaged(&lut[1], sizeof(int) * CDEPTH);
    err = cudaMallocManaged(&lut[2], sizeof(int) * CDEPTH);

    err = cudaMallocManaged(&maximum_1, sizeof(float));
    err = cudaMallocManaged(&minimum_1, sizeof(float));
    err = cudaMallocManaged(&maximum_2, sizeof(float));
    err = cudaMallocManaged(&minimum_2, sizeof(float));

    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);
    err = cudaStreamCreate(&s3);
    err = cudaStreamCreate(&s4);
    err = cudaStreamCreate(&s5);
}

void ImagePipeline::init(unsigned char* input_image, int channel) {
    gaussian_kernel(kernel_small, kernel_small_diameter, kernel_small_variance);
    gaussian_kernel(kernel_large, kernel_large_diameter, kernel_large_variance);
    gaussian_kernel(kernel_unsharpen, kernel_unsharpen_diameter, kernel_unsharpen_variance);
    *maximum_1 = 0;
    *minimum_1 = 0;
    *maximum_2 = 0;
    *minimum_2 = 0;

    // Initialize LUTs;
    lut_b(lut[0]);
    lut_g(lut[1]);
    lut_r(lut[2]);

    // Copy input data on the GPU managed memory;
    for (int i = 0; i < image_width * image_width; i++) {
        image[i] = int(input_image[black_and_white ? i : (i * 3 + channel)]);
    }
    cudaDeviceSynchronize();
}

void ImagePipeline::execute_sync(int channel) {

    if (pascalGpu && do_prefetch) {
        cudaMemPrefetchAsync(image, sizeof(int) * image_width * image_width, 0, 0);
        cudaMemPrefetchAsync(image2, sizeof(float) * image_width * image_width, 0, 0);
        cudaMemPrefetchAsync(image3, sizeof(int) * image_width * image_width, 0, 0);
        cudaMemPrefetchAsync(image_unsharpen, sizeof(float) * image_width * image_width, 0, 0);
        cudaMemPrefetchAsync(mask_small, sizeof(float) * image_width * image_width, 0, 0);
        cudaMemPrefetchAsync(mask_large, sizeof(float) * image_width * image_width, 0, 0);
        cudaMemPrefetchAsync(blurred_small, sizeof(float) * image_width * image_width, 0, 0);
        cudaMemPrefetchAsync(blurred_large, sizeof(float) * image_width * image_width, 0, 0);
        cudaMemPrefetchAsync(blurred_unsharpen, sizeof(float) * image_width * image_width, 0, 0);
    }
    // Blur - Small;
    gaussian_blur<<<grid_size_2d, block_size_2d, kernel_small_diameter * kernel_small_diameter * sizeof(float)>>>(image, blurred_small, image_width, image_width, kernel_small, kernel_small_diameter);
    cudaDeviceSynchronize();
    // Blur - Large;
    gaussian_blur<<<grid_size_2d, block_size_2d, kernel_large_diameter * kernel_large_diameter * sizeof(float)>>>(image, blurred_large, image_width, image_width, kernel_large, kernel_large_diameter);
    cudaDeviceSynchronize();
    // Blur - Unsharpen;
    gaussian_blur<<<grid_size_2d, block_size_2d, kernel_unsharpen_diameter * kernel_unsharpen_diameter * sizeof(float)>>>(image, blurred_unsharpen, image_width, image_width, kernel_unsharpen, kernel_unsharpen_diameter);
    cudaDeviceSynchronize();
    // Sobel filter (edge detection);
    sobel<<<grid_size_2d, block_size_2d>>>(blurred_small, mask_small, image_width, image_width);
    cudaDeviceSynchronize();
    sobel<<<grid_size_2d, block_size_2d>>>(blurred_large, mask_large, image_width, image_width);
    cudaDeviceSynchronize();
    // Ensure that the output of Sobel is in [0, 1];
    maximum_kernel<<<grid_size_1d, block_size_1d>>>(mask_small, maximum_1, image_width * image_width);
    cudaDeviceSynchronize();
    minimum_kernel<<<grid_size_1d, block_size_1d>>>(mask_small, minimum_1, image_width * image_width);
    cudaDeviceSynchronize();
    extend<<<grid_size_1d, block_size_1d>>>(mask_small, minimum_1, maximum_1, image_width * image_width, 1);
    cudaDeviceSynchronize();
    // Extend large edge detection mask, and normalize it;
    maximum_kernel<<<grid_size_1d, block_size_1d>>>(mask_large, maximum_2, image_width * image_width);
    cudaDeviceSynchronize();
    minimum_kernel<<<grid_size_1d, block_size_1d>>>(mask_large, minimum_2, image_width * image_width);
    cudaDeviceSynchronize();
    extend<<<grid_size_1d, block_size_1d>>>(mask_large, minimum_2, maximum_2, image_width * image_width, 5);
    cudaDeviceSynchronize();
    // Unsharpen;
    unsharpen<<<grid_size_1d, block_size_1d>>>(image, blurred_unsharpen, image_unsharpen, unsharpen_amount, image_width * image_width);
    cudaDeviceSynchronize();
    // Combine results;
    combine<<<grid_size_1d, block_size_1d>>>(image_unsharpen, blurred_large, mask_large, image2, image_width * image_width);
    cudaDeviceSynchronize();
    combine_lut<<<grid_size_1d, block_size_1d>>>(image2, blurred_small, mask_small, image3, image_width * image_width, lut[channel]);

    cudaDeviceSynchronize();
}

void ImagePipeline::execute_async(int channel) {
   
    if (!pascalGpu || stream_attach) {
        cudaStreamAttachMemAsync(s1, blurred_small, 0);
        cudaStreamAttachMemAsync(s1, mask_small, 0);
        cudaStreamAttachMemAsync(s2, blurred_large, 0);
        cudaStreamAttachMemAsync(s2, mask_large, 0);
        cudaStreamAttachMemAsync(s2, image2, 0);
        cudaStreamAttachMemAsync(s3, blurred_unsharpen, 0);
        cudaStreamAttachMemAsync(s3, image_unsharpen, 0);
        cudaStreamAttachMemAsync(s1, image3, 0);
    }
    if (pascalGpu && do_prefetch) {
        cudaMemPrefetchAsync(image, sizeof(int) * image_width * image_width, 0, s1);
        cudaMemPrefetchAsync(image2, sizeof(float) * image_width * image_width, 0, s2);
        cudaMemPrefetchAsync(image3, sizeof(int) * image_width * image_width, 0, s1);
        cudaMemPrefetchAsync(image_unsharpen, sizeof(float) * image_width * image_width, 0, s3);
        cudaMemPrefetchAsync(mask_small, sizeof(float) * image_width * image_width, 0, s1);
        cudaMemPrefetchAsync(mask_large, sizeof(float) * image_width * image_width, 0, s2);
        cudaMemPrefetchAsync(blurred_small, sizeof(float) * image_width * image_width, 0, s1);
        cudaMemPrefetchAsync(blurred_large, sizeof(float) * image_width * image_width, 0, s2);
        cudaMemPrefetchAsync(blurred_unsharpen, sizeof(float) * image_width * image_width, 0, s3);
    }
    // Blur - Small;
    gaussian_blur<<<grid_size_2d, block_size_2d, kernel_small_diameter * kernel_small_diameter * sizeof(float), s1>>>(image, blurred_small, image_width, image_width, kernel_small, kernel_small_diameter);
    // Blur - Large;
    gaussian_blur<<<grid_size_2d, block_size_2d, kernel_large_diameter * kernel_large_diameter * sizeof(float), s2>>>(image, blurred_large, image_width, image_width, kernel_large, kernel_large_diameter);
    // Blur - Unsharpen;
    gaussian_blur<<<grid_size_2d, block_size_2d, kernel_unsharpen_diameter * kernel_unsharpen_diameter * sizeof(float), s3>>>(image, blurred_unsharpen, image_width, image_width, kernel_unsharpen, kernel_unsharpen_diameter);
    // Sobel filter (edge detection);
    sobel<<<grid_size_2d, block_size_2d, 0, s1>>>(blurred_small, mask_small, image_width, image_width);
    sobel<<<grid_size_2d, block_size_2d, 0, s2>>>(blurred_large, mask_large, image_width, image_width);

    // Max-min + combine to normalize Sobel on small mask;
    cudaEvent_t e_ss, e_min1;
    cudaEventCreate(&e_ss);
    cudaEventCreate(&e_min1);
    cudaEventRecord(e_ss, s1);  // Wait end of Sobel on small mask; 
    maximum_kernel<<<grid_size_1d, block_size_1d, 0, s1>>>(mask_small, maximum_1, image_width * image_width);
    cudaStreamWaitEvent(s4, e_ss, 0);
    minimum_kernel<<<grid_size_1d, block_size_1d, 0, s4>>>(mask_small, minimum_1, image_width * image_width);
    cudaEventRecord(e_min1, s4);  
    cudaStreamWaitEvent(s1, e_min1, 0);  // Wait min;
    extend<<<grid_size_1d, block_size_1d, 0, s1>>>(mask_small, minimum_1, maximum_1, image_width * image_width, 1);
    
    // Max-min + combine to normalize Sobel on large mask;
    cudaEvent_t e_sl, e_min2;
    cudaEventCreate(&e_sl);
    cudaEventCreate(&e_min2);
    cudaEventRecord(e_sl, s2);
    maximum_kernel<<<grid_size_1d, block_size_1d, 0, s2>>>(mask_large, maximum_2, image_width * image_width);
    cudaStreamWaitEvent(s5, e_sl, 0);  // Wait end of Sobel on large mask; 
    minimum_kernel<<<grid_size_1d, block_size_1d, 0, s5>>>(mask_large, minimum_2, image_width * image_width);
    cudaEventRecord(e_min2, s5);  
    cudaStreamWaitEvent(s2, e_min2, 0);  // Wait min;
    extend<<<grid_size_1d, block_size_1d, 0, s2>>>(mask_large, minimum_2, maximum_2, image_width * image_width, 5);

    // Unsharpen;
    unsharpen<<<grid_size_1d, block_size_1d, 0, s3>>>(image, blurred_unsharpen, image_unsharpen, unsharpen_amount, image_width * image_width);

    // Combine results;
    cudaEvent_t e_un, e_co;
    cudaEventCreate(&e_un);
    cudaEventCreate(&e_co);
    cudaEventRecord(e_un, s3);
    cudaStreamWaitEvent(s2, e_un, 0);
    combine<<<grid_size_1d, block_size_1d, 0, s2>>>(image_unsharpen, blurred_large, mask_large, image2, image_width * image_width);
    cudaEventRecord(e_co, s2);
    cudaStreamWaitEvent(s1, e_co, 0);
    if (!pascalGpu || stream_attach) {
        cudaStreamAttachMemAsync(s1, image2, 0);
    }
    if (pascalGpu && do_prefetch) {
        cudaMemPrefetchAsync(image3, image_width * image_width * sizeof(float), 0, s1);
    }
    combine_lut<<<grid_size_1d, block_size_1d, 0, s1>>>(image2, blurred_small, mask_small, image3, image_width * image_width, lut[channel]);

    cudaStreamSynchronize(s1);
}

//////////////////////////////
// Utility functions /////////
//////////////////////////////

std::string ImagePipeline::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(image3[0]);
    } else {
        std::string res = "[";
        for (int j = 0; j < 10; j++) {
            res += std::to_string(image3[j]) + ", ";
        }
        return res + ", ...]";
    }
}

//////////////////////////////
// Main execution ////////////
//////////////////////////////

void ImagePipeline::run_inner(unsigned char* input_image, int channel) {
    auto start_tot = clock_type::now();
    auto start_tmp = clock_type::now();
    auto end_tmp = clock_type::now();

    // Allocation;
    start_tmp = clock_type::now();
    alloc();
    end_tmp = clock_type::now();
    if (debug && err) std::cout << "error=" << err << std::endl;
    if (debug) std::cout << "allocation time=" << chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count() / 1000 << " ms" << std::endl;

    // Initialization;
    start_tmp = clock_type::now();
    init(input_image, channel);
    end_tmp = clock_type::now();
    if (debug && err) std::cout << "error=" << err << std::endl;
    if (debug) std::cout << "initialization time=" << chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count() / 1000 << " ms" << std::endl;

    // Execution;
    start_tmp = clock_type::now();
    switch (policy) {
        case Policy::Sync:
            execute_sync(channel);
            break;
        default:
            execute_async(channel);
    }
    if (debug && err) std::cout << "  error=" << err << std::endl;
    end_tmp = clock_type::now();
    auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
    if (debug) {
        std::cout << "  result=" << print_result() << std::endl;
        std::cout << "  execution=" << float(exec_time) / 1000 << " ms" << std::endl;
    }

    // Copy back data;
    start_tmp = clock_type::now();
    for (int i = 0; i < image_width * image_width; i++) {
        input_image[black_and_white ? i : (i * 3 + channel)] = (unsigned char) (image3[i]);
    }
    end_tmp = clock_type::now();
    auto write_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
    if (debug) std::cout << "writeback time=" << write_time / 1e6 << " sec" << std::endl;

    // End of the coputation;
    auto end_time = chrono::duration_cast<chrono::microseconds>(clock_type::now() - start_tot).count();
    if (debug) std::cout << "\ntotal processing time=" << end_time / 1e6 << " sec" << std::endl;
}

void ImagePipeline::run(unsigned char* input_image) {

    int deviceCount = 1;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount >= 4) {
        cudaSetDevice(3);
    }

    for (int channel = 0; channel < (black_and_white ? 1 : 3); channel++) {
        // Access individual channels;
        run_inner(input_image, channel);
    }
}

