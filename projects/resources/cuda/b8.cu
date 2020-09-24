#include "b8.cuh"

//////////////////////////////
//////////////////////////////

extern "C" __global__ void gaussian_blur(const float *image, float *result, int rows, int cols, const float *kernel, int diameter) {
    extern __shared__ float kernel_local[];
    for (int i = threadIdx.x; i < diameter; i += blockDim.x) {
        for (int j = threadIdx.y; j < diameter; j += blockDim.y) {
            kernel_local[i * diameter + j] = kernel[i * diameter + j];
        }
    }
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols; j += blockDim.y * gridDim.y) {
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

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols; j += blockDim.y * gridDim.y) {
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

__device__ float atomicMinf(float *address, float val) {
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }
    return __int_as_float(old);
}

__device__ float atomicMaxf(float *address, float val) {
    int *address_as_int = (int *)address;
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

extern "C" __global__ void maximum_kernel(const float *in, float *out, int N) {
    int warp_size = 32;
    float maximum = -1000;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        maximum = max(maximum, in[i]);
    }
    maximum = warp_reduce_max(maximum);        // Obtain the max of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMaxf(out, maximum);              // The first thread in the warp updates the output;
}

extern "C" __global__ void minimum_kernel(const float *in, float *out, int N) {
    int warp_size = 32;
    float minimum = 1000;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        minimum = min(minimum, in[i]);
    }
    minimum = warp_reduce_min(minimum);        // Obtain the min of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMinf(out, minimum);              // The first thread in the warp updates the output;
}

extern "C" __global__ void extend(float *x, const float *minimum, const float *maximum, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float res_tmp = 5 * (x[i] - *minimum) / (*maximum - *minimum);
        x[i] = res_tmp > 1 ? 1 : res_tmp;
    }
}

extern "C" __global__ void unsharpen(const float *x, const float *y, float *res, float amount, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float res_tmp = x[i] * (1 + amount) - y[i] * amount;
        res_tmp = res_tmp > 1 ? 1 : res_tmp;
        res[i] = res_tmp < 0 ? 0 : res_tmp;
    }
}

extern "C" __global__ void combine(const float *x, const float *y, const float *mask, float *res, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = x[i] * mask[i] + y[i] * (1 - mask[i]);
    }
}

extern "C" __global__ void reset_image(float *x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] = 0.0;
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark8::alloc() {
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

    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);
    err = cudaStreamCreate(&s3);
    err = cudaStreamCreate(&s4);
    err = cudaStreamCreate(&s5);
}

void Benchmark8::init() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            image[i * N + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    gaussian_kernel(kernel_small, kernel_small_diameter, 1);
    gaussian_kernel(kernel_large, kernel_large_diameter, 10);
    gaussian_kernel(kernel_unsharpen, kernel_unsharpen_diameter, 5);
}

void Benchmark8::reset() {
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         image3[i * N + j] = 0;
    //     }
    // }
    memset(image3, 0, N * N * sizeof(float));
    *maximum = 0;
    *minimum = 0;
    reset_image<<<num_blocks, block_size_1d>>>(image3, N * N);
    cudaDeviceSynchronize();
}

void Benchmark8::execute_sync(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks, num_blocks);
    dim3 grid_size_2(num_blocks / 2, num_blocks / 2);

    gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_small_diameter * kernel_small_diameter * sizeof(float)>>>(image, blurred_small, N, N, kernel_small, kernel_small_diameter);
    cudaDeviceSynchronize();

    gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_large_diameter * kernel_large_diameter * sizeof(float)>>>(image, blurred_large, N, N, kernel_large, kernel_large_diameter);
    cudaDeviceSynchronize();

    gaussian_blur<<<grid_size_2, block_size_2d_dim, kernel_unsharpen_diameter * kernel_unsharpen_diameter * sizeof(float)>>>(image, blurred_unsharpen, N, N, kernel_unsharpen, kernel_unsharpen_diameter);
    cudaDeviceSynchronize();

    sobel<<<grid_size_2, block_size_2d_dim>>>(blurred_small, mask_small, N, N);
    cudaDeviceSynchronize();

    sobel<<<grid_size_2, block_size_2d_dim>>>(blurred_large, mask_large, N, N);
    cudaDeviceSynchronize();

    maximum_kernel<<<num_blocks, block_size_1d>>>(mask_large, maximum, N * N);
    cudaDeviceSynchronize();

    minimum_kernel<<<num_blocks, block_size_1d>>>(mask_large, minimum, N * N);
    cudaDeviceSynchronize();

    extend<<<num_blocks, block_size_1d>>>(mask_large, minimum, maximum, N * N);
    cudaDeviceSynchronize();

    unsharpen<<<num_blocks, block_size_1d>>>(image, blurred_unsharpen, image_unsharpen, 0.5, N * N);
    cudaDeviceSynchronize();

    combine<<<num_blocks, block_size_1d>>>(image_unsharpen, blurred_large, mask_large, image2, N * N);
    cudaDeviceSynchronize();

    combine<<<num_blocks, block_size_1d>>>(image2, blurred_small, mask_small, image3, N * N);

    // Extra
    // combine<<<num_blocks, block_size_1d>>>(blurred_small, blurred_large, blurred_unsharpen, image3, N * N);

    cudaDeviceSynchronize();
}

void Benchmark8::execute_async(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks, num_blocks);
    int nb = num_blocks / 2;
    dim3 grid_size_2(nb, nb);

    cudaStreamAttachMemAsync(s1, blurred_small, 0);
    cudaStreamAttachMemAsync(s1, mask_small, 0);
    cudaStreamAttachMemAsync(s2, blurred_large, 0);
    cudaStreamAttachMemAsync(s2, mask_large, 0);
    cudaStreamAttachMemAsync(s2, image2, 0);
    cudaStreamAttachMemAsync(s3, blurred_unsharpen, 0);
    cudaStreamAttachMemAsync(s3, image_unsharpen, 0);
    cudaStreamAttachMemAsync(s1, image3, 0);

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
}

void Benchmark8::execute_cudagraph(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks, num_blocks);
    int nb = num_blocks / 2;
    dim3 grid_size_2(nb, nb);
    if (iter == 0) {
        cudaEvent_t ef;
        cudaEventCreate(&ef);
        cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
        cudaEventRecord(ef, s1);
        cudaStreamWaitEvent(s2, ef, 0);
        cudaStreamWaitEvent(s3, ef, 0);
        cudaStreamWaitEvent(s4, ef, 0);
        cudaStreamWaitEvent(s5, ef, 0);

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

        combine<<<num_blocks, block_size_1d, 0, s1>>>(image2, blurred_small, mask_small, image3, N * N);

        cudaStreamEndCapture(s1, &graph);
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }
    cudaGraphLaunch(graphExec, s1);
    err = cudaStreamSynchronize(s1);
}

void Benchmark8::execute_cudagraph_manual(int iter) {}

std::string Benchmark8::print_result(bool short_form) {
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
