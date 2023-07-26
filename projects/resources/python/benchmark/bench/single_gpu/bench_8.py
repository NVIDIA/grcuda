# Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# coding=utf-8
import polyglot
from java.lang import System 
import numpy as np
from random import randint, seed

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D, DEFAULT_BLOCK_SIZE_2D
from benchmark_result import BenchmarkResult

##############################
##############################

NUM_THREADS_PER_BLOCK_2D = 8
NUM_THREADS_PER_BLOCK = 32
WARP_SIZE = 32

GAUSSIAN_BLUR = """
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
"""


SOBEL = """

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
"""

EXTEND_MASK = """
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

extern "C" __global__ void maximum(float *in, float* out, int N) {
    int warp_size = 32;
    float maximum = -1000;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) { 
        maximum = max(maximum, in[i]);
    }
    maximum = warp_reduce_max(maximum); // Obtain the max of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMaxf(out, maximum); // The first thread in the warp updates the output;
}

extern "C" __global__ void minimum(float *in, float* out, int N) {
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
"""

UNSHARPEN = """
extern "C" __global__ void unsharpen(float *x, float *y, float *res, float amount, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        float res_tmp = x[i] * (1 + amount) - y[i] * amount;
        res_tmp = res_tmp > 1 ? 1 : res_tmp;
        res[i] = res_tmp < 0 ? 0 : res_tmp;
    }
}
"""

COMBINE = """
extern "C" __global__ void combine(const float *x, const float *y, const float *mask, float *res, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        res[i] = x[i] * mask[i] + y[i] * (1 - mask[i]);
    }
}
"""

RESET = """
extern "C" __global__ void reset(float *x, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        x[i] = 0.0;
    }
}
"""

##############################
##############################


class Benchmark8(Benchmark):
    """
    Compute an image processing pipeline in which we sharpen an image and combine it
    with copies that have been blurred at low and medium frequencies. The result is an image sharper on the edges,
    and softer everywhere else: this filter is common, for example, in portrait retouching, where a photographer desires
    to enhance the clarity of facial features while smoothing the subject' skin and the background;

    The input is a random square single-channel image with floating-point values between 0 and 1, with side of length size.

    BLUR(image,blur1) ─> SOBEL(blur1,mask1) ───────────────────────────────────────────────────────────────────────────────┐
    BLUR(image,blur2) ─> SOBEL(blur2,mask2) ┬─> MAX(mask2) ──┬─> EXTEND(mask2) ──┐                                         │
                                            └─> MIN(mask2) ──┘                   │                                         │
    SHARPEN(image,blur3) ─> UNSHARPEN(image,blur3,sharpened) ────────────────────┴─> COMBINE(sharpened,blur2,mask2,image2) ┴─> COMBINE(image2,blur1,mask1,image3)
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b8", benchmark, nvprof_profile)
        self.size = 0

        self.image = None
        self.image2 = None
        self.image3 = None

        self.blurred_small = None
        self.mask_small = None
        self.kernel_small = None
        self.kernel_small_diameter = 3
        self.kernel_small_variance = 1

        self.blurred_large = None
        self.mask_large = None
        self.kernel_large = None
        self.kernel_large_diameter = 5
        self.kernel_large_variance = 10
        self.maximum = None
        self.minimum = None
        self.reset = None

        self.blurred_unsharpen = None
        self.image_unsharpen = None
        self.kernel_unsharpen = None
        self.kernel_unsharpen_diameter = 3
        self.kernel_unsharpen_variance = 5
        self.unsharpen_amount = 0.5

        # self.image_cpu = None
        self.kernel_small_cpu = None
        self.kernel_large_cpu = None
        self.kernel_unsharpen_cpu = None

        self.cpu_result = None
        self.gpu_result = None

        self.num_blocks_per_processor = self.num_blocks # 12  # 32

        self.block_size_1d = DEFAULT_BLOCK_SIZE_1D
        self.block_size_2d = DEFAULT_BLOCK_SIZE_2D

        self.gaussian_blur_kernel = None
        self.sobel_kernel = None
        self.extend_kernel = None
        self.unsharpen_kernel = None
        self.combine_mask_kernel = None
        self.maximum_kernel = None
        self.minimum_kernel = None

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.block_size_1d = block_size["block_size_1d"]
        self.block_size_2d = block_size["block_size_2d"]

        # Allocate vectors;
        self.image = polyglot.eval(language="grcuda", string=f"float[{size * size}]")
        self.image2 = polyglot.eval(language="grcuda", string=f"float[{size}][{size}]")
        self.image3 = polyglot.eval(language="grcuda", string=f"float[{size}][{size}]")

        self.kernel_small = polyglot.eval(language="grcuda", string=f"float[{self.kernel_small_diameter}][{self.kernel_small_diameter}]")
        self.kernel_large = polyglot.eval(language="grcuda", string=f"float[{self.kernel_large_diameter}][{self.kernel_large_diameter}]")
        self.kernel_unsharpen = polyglot.eval(language="grcuda", string=f"float[{self.kernel_unsharpen_diameter}][{self.kernel_unsharpen_diameter}]")
        self.maximum = polyglot.eval(language="grcuda", string=f"float[1]")
        self.minimum = polyglot.eval(language="grcuda", string=f"float[1]")

        self.mask_small = polyglot.eval(language="grcuda", string=f"float[{size}][{size}]")
        self.mask_large = polyglot.eval(language="grcuda", string=f"float[{size}][{size}]")
        self.image_unsharpen = polyglot.eval(language="grcuda", string=f"float[{size}][{size}]")

        self.blurred_small = polyglot.eval(language="grcuda", string=f"float[{size}][{size}]")
        self.blurred_large = polyglot.eval(language="grcuda", string=f"float[{size}][{size}]")
        self.blurred_unsharpen = polyglot.eval(language="grcuda", string=f"float[{size}][{size}]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.gaussian_blur_kernel = build_kernel(GAUSSIAN_BLUR, "gaussian_blur", "const pointer, pointer, sint32, sint32, const pointer, sint32")
        self.sobel_kernel = build_kernel(SOBEL, "sobel", "pointer, pointer, sint32, sint32")
        self.extend_kernel = build_kernel(EXTEND_MASK, "extend", "pointer, const pointer, const pointer, sint32")
        self.maximum_kernel = build_kernel(EXTEND_MASK, "maximum", "const pointer, pointer, sint32")
        self.minimum_kernel = build_kernel(EXTEND_MASK, "minimum", "const pointer, pointer, sint32")
        self.unsharpen_kernel = build_kernel(UNSHARPEN, "unsharpen", "pointer, pointer, pointer, float, sint32")
        self.combine_mask_kernel = build_kernel(COMBINE, "combine", "const pointer, const pointer, const pointer, pointer, sint32")
        self.reset_kernel = build_kernel(RESET, "reset", "pointer, sint32")
        self.initialize_rand = polyglot.eval(language="js", string="x => { for (let i = 0; i < x.length; i++) { x[i] = Math.random() }}")

    @time_phase("initialization")
    def init(self):

        def gaussian_kernel(diameter, sigma):
            kernel = np.zeros((diameter, diameter))
            mean = diameter / 2
            sum_tmp = 0
            for x in range(diameter):
                for y in range(diameter):
                    kernel[x, y] = np.exp(-0.5 * ((x - mean) ** 2 + (y - mean) ** 2) / sigma ** 2)
                    sum_tmp += kernel[x, y]
            for x in range(diameter):
                for y in range(diameter):
                    kernel[x, y] /= sum_tmp
            return kernel

        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)

        # Create a random image;
        self.initialize_rand(self.image)
        self.gpu_result = [[0.0] * self.size for _ in range(self.size)]
        # self.image_cpu = np.random.rand(self.size, self.size).astype(np.float32)  # Create here the image used for validation;
        # self.image.copyFrom(int(np.int64(self.image_cpu.ctypes.data)), len(self.image))
        # self.gpu_result = np.zeros((self.size, self.size))
        self.kernel_small_cpu = gaussian_kernel(self.kernel_small_diameter, self.kernel_small_variance)
        self.kernel_large_cpu = gaussian_kernel(self.kernel_large_diameter, self.kernel_large_variance)
        self.kernel_unsharpen_cpu = gaussian_kernel(self.kernel_unsharpen_diameter, self.kernel_unsharpen_variance)
        for i in range(self.kernel_small_diameter):
            for j in range(self.kernel_small_diameter):
                self.kernel_small[i][j] = float(self.kernel_small_cpu[i, j])
        for i in range(self.kernel_large_diameter):
            for j in range(self.kernel_large_diameter):
                self.kernel_large[i][j] = float(self.kernel_large_cpu[i, j])
        for i in range(self.kernel_unsharpen_diameter):
            for j in range(self.kernel_unsharpen_diameter):
                self.kernel_unsharpen[i][j] = float(self.kernel_unsharpen_cpu[i, j])

    @time_phase("reset_result")
    def reset_result(self) -> None:
        # for i in range(self.size):
        #     for j in range(self.size):
        #         self.image3[i][j] = 0.0
        # self.image3.copyFrom(int(np.int64(self.image_cpu.ctypes.data)), len(self.image3))
        self.maximum[0] = 0.0
        self.minimum[0] = 0.0

    def execute(self) -> object:
        self.block_size_1d = self._block_size["block_size_1d"]
        self.block_size_2d = self._block_size["block_size_2d"]
        self.num_blocks_per_processor = self.num_blocks  # 12  # 32
        a = self.num_blocks_per_processor / 2

        start_comp = System.nanoTime()
        start = 0

        self.reset_kernel(self.num_blocks_per_processor, self.block_size_1d)(self.image3, 0)

        self.reset_kernel((a, a), (self.block_size_2d, self.block_size_2d))(self.image3, 0)

        # Blur - Small;
        self.execute_phase("blur_small",
                           self.gaussian_blur_kernel((a, a), (self.block_size_2d, self.block_size_2d), 4 * self.kernel_small_diameter**2),
                           self.image, self.blurred_small, self.size, self.size, self.kernel_small, self.kernel_small_diameter)

        # Blur - Large;
        self.execute_phase("blur_large",
                           self.gaussian_blur_kernel((a, a), (self.block_size_2d, self.block_size_2d), 4 * self.kernel_large_diameter**2),
                           self.image, self.blurred_large, self.size, self.size, self.kernel_large, self.kernel_large_diameter)

        # Blur - Unsharpen;
        self.execute_phase("blur_unsharpen",
                           self.gaussian_blur_kernel((a, a), (self.block_size_2d, self.block_size_2d), 4 * self.kernel_unsharpen_diameter**2),
                           self.image, self.blurred_unsharpen, self.size, self.size, self.kernel_unsharpen, self.kernel_unsharpen_diameter)

        # Sobel filter (edge detection);
        self.execute_phase("sobel_small",
                           self.sobel_kernel((a, a), (self.block_size_2d, self.block_size_2d)),
                           self.blurred_small, self.mask_small, self.size, self.size)

        self.execute_phase("sobel_large",
                           self.sobel_kernel((a, a), (self.block_size_2d, self.block_size_2d)),
                           self.blurred_large, self.mask_large, self.size, self.size)

        # Extend large edge detection mask;
        self.execute_phase("maximum",
                           self.maximum_kernel(self.num_blocks_per_processor, self.block_size_1d), self.mask_large, self.maximum, self.size**2)
        self.execute_phase("minimum",
                           self.minimum_kernel(self.num_blocks_per_processor, self.block_size_1d), self.mask_large, self.minimum, self.size**2)
        self.execute_phase("extend",
                           self.extend_kernel(self.num_blocks_per_processor, self.block_size_1d), self.mask_large, self.minimum, self.maximum, self.size**2)

        # Unsharpen;
        self.execute_phase("unsharpen",
                            self.unsharpen_kernel(self.num_blocks_per_processor, self.block_size_1d),
                            self.image, self.blurred_unsharpen, self.image_unsharpen, self.unsharpen_amount, self.size * self.size)

        # Combine results;
        self.execute_phase("combine",
                           self.combine_mask_kernel(self.num_blocks_per_processor, self.block_size_1d),
                           self.image_unsharpen, self.blurred_large, self.mask_large, self.image2, self.size * self.size)
        self.execute_phase("combine_2",
                           self.combine_mask_kernel(self.num_blocks_per_processor, self.block_size_1d),
                           self.image2, self.blurred_small, self.mask_small, self.image3, self.size * self.size)

        # Add a final sync step to measure the real computation time;
        if self.time_phases:
            start = System.nanoTime()
        tmp = self.image3[0][0]
        end = System.nanoTime()
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)

        # Compute GPU result;
        # for i in range(self.size):
        #     for j in range(self.size):
        #         self.gpu_result[i, j] = self.image3[i][j]

        self.gpu_result = sum(self.image3[-1])

        self.benchmark.add_to_benchmark("gpu_result", 0)
        if self.benchmark.debug:
            BenchmarkResult.log_message(
                f"\tgpu result: [" + ", ".join([f"{x:.4f}" for x in self.image3[0][:10]]) + "...]")

        return self.gpu_result

    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:

        sobel_filter_diameter = 3
        sobel_filter_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_filter_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        def sobel_filter(image):
            out = np.zeros(image.shape)
            rows, cols = image.shape
            radius = sobel_filter_diameter // 2

            for i in range(rows):
                for j in range(cols):
                    sum_gradient_x = 0
                    sum_gradient_y = 0
                    for x in range(-radius, radius + 1):
                        for y in range(-radius, radius + 1):
                            nx = x + i
                            ny = y + j
                            if (nx >= 0 and ny >= 0 and nx < rows and ny < cols):
                                gray_value_neigh = image[nx, ny]
                                gradient_x = sobel_filter_x[x + radius][y + radius]
                                gradient_y = sobel_filter_y[x + radius][y + radius]
                                sum_gradient_x += gray_value_neigh * gradient_x
                                sum_gradient_y += gray_value_neigh * gradient_y
                    out[i, j] = np.sqrt(sum_gradient_x ** 2 + sum_gradient_y ** 2)
            return out

        def gaussian_blur(image, kernel):
            out = np.zeros(image.shape)
            rows, cols = image.shape

            # Blur radius;
            diameter = kernel.shape[0]
            radius = diameter // 2

            # Flatten image and kernel;
            image_1d = image.reshape(-1)
            kernel_1d = kernel.reshape(-1)

            for i in range(rows):
                for j in range(cols):
                    sum_tmp = 0
                    for x in range(-radius, radius + 1):
                        for y in range(-radius, radius + 1):
                            nx = x + i
                            ny = y + j
                            if (nx >= 0 and ny >= 0 and nx < rows and ny < cols):
                                sum_tmp += kernel_1d[(x + radius) * diameter + (y + radius)] * image_1d[nx * cols + ny]
                    out[i, j] = sum_tmp
            return out

        def normalize(image):
            return (image - np.min(image)) / (np.max(image) - np.min(image))

        def truncate(image, minimum=0, maximum=1):
            out = image.copy()
            out[out < minimum] = minimum
            out[out > maximum] = maximum
            return out

        # Recompute the CPU result only if necessary;
        start = System.nanoTime()
        if self.current_iter == 0 or reinit:

            image_cpu = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(self.size):
                    image_cpu[i, j] = self.image[i * self.size + j]

            # Part 1: Small blur on medium frequencies;
            blurred_small = gaussian_blur(image_cpu, self.kernel_small_cpu)
            edges_small = sobel_filter(blurred_small)

            # Part 2: High blur on low frequencies;
            blurred_large = gaussian_blur(image_cpu, self.kernel_large_cpu)
            edges_large = sobel_filter(blurred_large)
            # Extend mask to cover a larger area;
            edges_large = normalize(edges_large) * 5
            edges_large[edges_large > 1] = 1

            # Part 3: Sharpen image;
            unsharpen = gaussian_blur(image_cpu, self.kernel_unsharpen_cpu)
            amount = 0.5
            sharpened = truncate(image_cpu * (1 + amount) - unsharpen * amount)

            # Part 4: Merge sharpened image and low frequencies;
            image2 = normalize(sharpened * edges_large + blurred_large * (1 - edges_large))

            # Part 5: Merge image and medium frequencies;
            self.cpu_result = image2 * edges_small + blurred_small * (1 - edges_small)

        cpu_time = System.nanoTime() - start

        # Compare GPU and CPU results;
        difference = sum(self.cpu_result[-1, :]) - gpu_result
        # difference = 0
        # for i in range(self.size):
        #     for j in range(self.size):
        #         difference += np.abs(self.cpu_result[i, j] - gpu_result[i, j])

        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", str(difference))
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: [" + ", ".join([f"{x:.4f}" for x in self.cpu_result[0, :10]])
                                        + "...]; " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")
