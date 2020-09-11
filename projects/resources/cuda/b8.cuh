
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
