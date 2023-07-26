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
from random import random, randint, seed, sample, uniform

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D, DEFAULT_BLOCK_SIZE_2D, DEFAULT_NUM_BLOCKS
from benchmark_result import BenchmarkResult

##############################
##############################

NUM_THREADS_PER_BLOCK_2D = 8
NUM_THREADS_PER_BLOCK = 32
WARP_SIZE = 32

CONV2D = """
extern "C" __global__ void conv2d(float *out, float *x, float *kernels, int N, int M, int L, int K, int k_out, int stride) {
    extern __shared__ float kernel_local[];
    int radius = K / 2;
    
    for (int m = 0; m < k_out; m++) {
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            for (int j = threadIdx.y; j < K; j += blockDim.y) {
                for (int l = 0; l < L; l++) {
                    kernel_local[l + L * (j + K * (i + K * m))] = kernels[l + L * (j  + K * (i + K * m))];
                }
            }
        }
    }
    __syncthreads();
    
   
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (int) ceilf((float) N / stride) - radius; i += blockDim.x * gridDim.x) {
        int out_index = M * i / stride;
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < (int) ceilf((float) M / stride) - radius; j += blockDim.y * gridDim.y) {
            for (int m = 0; m < k_out; m++) {
            // for (int m = blockIdx.z * blockDim.z + threadIdx.z; m < k_out; m += blockDim.z * gridDim.z) {
                float res = 0;
                int i_f = i * stride + radius;
                int j_f = j * stride + radius;
                for (int k_i = -radius; k_i <= radius; k_i++) {
                    for (int k_j = -radius; k_j <= radius; k_j++) {
                        int kernel_index = (k_j + radius + K * (k_i + radius + K * m));
                        for (int l = 0; l < L; l++) {                
                            int ni = i_f + k_i;
                            int nj = j_f + k_j;
                            res += kernel_local[l + L * kernel_index] * x[((ni * M) + nj) * L + l];
                        }
                    }
                }
                // Apply ReLU operator;
                out[m + k_out * (j + out_index)] = max(res, 0.0);
            }
        }
    }
}
"""

POOLING = """
extern "C" __global__ void mean_pooling(float *out, float *x, int N, int M, int L, int K, int stride) {
    int radius = K / 2;   
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (int) ceilf((float) N / stride) - radius; i += blockDim.x * gridDim.x) {
        int out_index = M * i / stride;
        int i_f = i * stride + radius;
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < (int) ceilf((float) M / stride) - radius; j += blockDim.y * gridDim.y) {
            int j_f = j * stride + radius;
            for (int l = blockIdx.z * blockDim.z + threadIdx.z; l < L; l += blockDim.z * gridDim.z) {
                float res = 0;
                for (int k_i = -radius; k_i <= radius; k_i++) {
                    int ni = i_f + k_i;
                    for (int k_j = -radius; k_j <= radius; k_j++) {
                        int nj = j_f + k_j;
                        res += x[((ni * M) + nj) * L + l];
                    }
                }
                // Apply mean operator;
                out[l + L * (j + out_index)] = res / (K * K);
            }
        }
    }
}
"""

GAP = """
extern "C" __global__ void gap(float *out, float *x, int N, int M, int L) {
    extern __shared__ float out_local[];
    for(int i = threadIdx.x; i < L; i += blockDim.x) {
        out_local[i] = 0;
    }
    __syncthreads();
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < M; j += blockDim.y * gridDim.y) {
            for (int l = 0; l < L; l++) {   
                atomicAdd(out_local + l, x[l + L * (j + M * i)]);
            }
        }
    }
    __syncthreads();
    for(int l = threadIdx.x; l < L; l += blockDim.x) {
        atomicAdd(out + l, out_local[l] / (M * N));
    }
}
"""

DOT_PRODUCT = """
__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void dot_product(const float *x, const float *y, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] * y[i];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}
"""

CONCAT = """
extern "C" __global__ void concat(float *z, const float *x, const float *y, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        z[i] = x[i];
        z[i + n] = y[i];
    }
}
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

##############################
##############################


class Benchmark10(Benchmark):
    """
    Compute a convolutional neural network that takes 2 images as inputs, computes their low-dimensional embeddings,
    concatenate them and apply a dense classifier. It can represent, for example, a network that detects if 2 images contain the same object;

    CONV(x) ─> CONV(x1) ─> GAP(x2) ──┬─> CONCAT(x2, y2) ─> DENSE(z)
    CONV(y) ─> CONV(y1) ─> GAP(y2) ──┘
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b10", benchmark, nvprof_profile)
        self.size = 0

        self.x = None
        self.y = None
        self.x_cpu = None
        self.y_cpu = None

        self.kernel_1 = None
        self.kernel_2 = None
        self.kernel_3 = None
        self.kernel_4 = None
        self.channels = 1
        self.K = 3
        self.kn1 = 8
        self.kn2 = 16
        self.stride = 2
        self.pooling = 5

        self.x1 = None
        self.x2 = None
        self.x11 = None
        self.y11 = None
        self.x3 = None
        self.y1 = None
        self.y2 = None
        self.y3 = None
        self.z = None
        self.res = None
        self.dense_weights = None

        self.cpu_result = None
        self.gpu_result = None

        self.num_blocks_per_processor = self.num_blocks

        self.block_size_1d = DEFAULT_BLOCK_SIZE_1D
        self.block_size_2d = DEFAULT_BLOCK_SIZE_2D

        self.conv2d_kernel = None
        self.gap_kernel = None
        self.concat_kernel = None
        self.dp_kernel = None
        self.pooling_kernel = None

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.block_size_1d = block_size["block_size_1d"]
        self.block_size_2d = block_size["block_size_2d"]

        self.gpu_result = 0.0

        # Allocate vectors;
        self.x = polyglot.eval(language="grcuda", string=f"float[{size * size * self.channels}]")
        self.x1 = polyglot.eval(language="grcuda", string=f"float[{(size // self.stride) * (size // self.stride) * self.kn1}]")
        self.x11 = polyglot.eval(language="grcuda", string=f"float[{(size // self.stride // self.pooling) * (size // self.stride // self.pooling) * self.kn1}]")
        self.x2 = polyglot.eval(language="grcuda", string=f"float[{(size // self.stride // self.pooling // self.stride) * (size // self.stride // self.pooling // self.stride) * self.kn2}]")
        self.x3 = polyglot.eval(language="grcuda", string=f"float[{self.kn2}]")
        self.y = polyglot.eval(language="grcuda", string=f"float[{size * size * self.channels}]")
        self.y1 = polyglot.eval(language="grcuda", string=f"float[{(size // self.stride) * (size // self.stride) * self.kn1}]")
        self.y11 = polyglot.eval(language="grcuda", string=f"float[{(size // self.stride // self.pooling) * (size // self.stride // self.pooling) * self.kn1}]")
        self.y2 = polyglot.eval(language="grcuda", string=f"float[{(size // self.stride // self.pooling // self.stride) * (size // self.stride // self.pooling // self.stride) * self.kn2}]")
        self.y3 = polyglot.eval(language="grcuda", string=f"float[{self.kn2}]")
        self.kernel_1 = polyglot.eval(language="grcuda", string=f"float[{self.kn1 * self.K * self.K * self.channels}]")
        self.kernel_2 = polyglot.eval(language="grcuda", string=f"float[{self.kn1 * self.K * self.K * self.kn2}]")
        self.kernel_3 = polyglot.eval(language="grcuda", string=f"float[{self.kn1 * self.K * self.K * self.channels}]")
        self.kernel_4 = polyglot.eval(language="grcuda", string=f"float[{self.kn1 * self.K * self.K * self.kn2}]")
        self.z = polyglot.eval(language="grcuda", string=f"float[{len(self.y2) * 2}]")
        self.dense_weights = polyglot.eval(language="grcuda", string=f"float[{len(self.z)}]")
        self.res = polyglot.eval(language="grcuda", string=f"float[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.conv2d_kernel = build_kernel(CONV2D, "conv2d", "pointer, pointer, const pointer, sint32, sint32, sint32, sint32, sint32, sint32")
        self.pooling_kernel = build_kernel(POOLING, "mean_pooling", "pointer, const pointer, sint32, sint32, sint32, sint32, sint32")
        self.gap_kernel = build_kernel(GAP, "gap", "pointer, pointer, sint32, sint32, sint32")
        self.concat_kernel = build_kernel(CONCAT, "concat", "pointer, const pointer, const pointer, sint32")
        self.dp_kernel = build_kernel(DOT_PRODUCT, "dot_product", "const pointer, const pointer, pointer, sint32")

    @time_phase("initialization")
    def init(self):

        self.random_seed = 10 # randint(0, 10000000)
        seed(self.random_seed)

        # Random weights;
        for i in range(len(self.kernel_1)):
            self.kernel_1[i] = uniform(-1, 1)
            self.kernel_3[i] = uniform(-1, 1)
        for i in range(len(self.kernel_2)):
            self.kernel_2[i] = uniform(-1, 1)
            self.kernel_4[i] = uniform(-1, 1)

        for i in range(len(self.dense_weights)):
            self.dense_weights[i] = uniform(-1, 1) / len(self.dense_weights)

        # Create random images. Leave it for last so that we can re-create identical random weights from the same seed;
        self.x_cpu = [0] * len(self.x)
        self.y_cpu = [0] * len(self.y)
        for i in range(len(self.x_cpu)):
            self.x_cpu[i] = random()
            self.y_cpu[i] = random()

    @time_phase("reset_result")
    def reset_result(self) -> None:
        self.gpu_result = 0.0
        self.res[0] = 0.0
        for i in range(len(self.x_cpu)):
            self.x[i] = self.x_cpu[i]
            self.y[i] = self.y_cpu[i]

    def execute(self) -> object:
        self.num_blocks_per_processor = self.num_blocks
        self.block_size_1d = self._block_size["block_size_1d"]
        self.block_size_2d = self._block_size["block_size_2d"]
        start_comp = System.nanoTime()
        start = 0

        a = self.num_blocks_per_processor / 2
        # Convolutions;
        self.execute_phase("conv_x1",
                           self.conv2d_kernel((a, a), (self.block_size_2d, self.block_size_2d), 4 * (self.K ** 2) * self.kn1 * self.channels),
                           self.x1, self.x, self.kernel_1, self.size, self.size, self.channels, self.K, self.kn1, self.stride)
        self.execute_phase("conv_y1",
                           self.conv2d_kernel((a, a), (self.block_size_2d, self.block_size_2d), 4 * (self.K ** 2) * self.kn1 * self.channels),
                           self.y1, self.y, self.kernel_3, self.size, self.size, self.channels, self.K, self.kn1, self.stride)
        # Pooling;
        self.execute_phase("pool_x1",
                           self.pooling_kernel((a / 2, a / 2, a / 2), (self.block_size_2d / 2, self.block_size_2d / 2, self.block_size_2d / 2)),
                           self.x11, self.x1, self.size // self.stride, self.size // self.stride, self.kn1, self.pooling, self.pooling)
        self.execute_phase("pool_y1",
                           self.pooling_kernel((a / 2, a / 2, a / 2), (self.block_size_2d / 2, self.block_size_2d / 2, self.block_size_2d / 2)),
                           self.y11, self.y1, self.size // self.stride, self.size // self.stride, self.kn1, self.pooling, self.pooling)
        # Other convolutions;
        self.execute_phase("conv_x2",
                           self.conv2d_kernel((a, a), (self.block_size_2d, self.block_size_2d), 4 * (self.K ** 2) * self.kn1 * self.kn2),
                           self.x2, self.x11, self.kernel_2, self.size // self.stride // self.pooling, self.size // self.stride // self.pooling, self.kn1, self.K, self.kn2, self.stride)
        self.execute_phase("conv_y2",
                           self.conv2d_kernel((a, a), (self.block_size_2d, self.block_size_2d), 4 * (self.K ** 2) * self.kn1 * self.kn2),
                           self.y2, self.y11, self.kernel_4, self.size // self.stride // self.pooling, self.size // self.stride // self.pooling, self.kn1, self.K, self.kn2, self.stride)

        # Global average pooling;
        # self.execute_phase("gap_x",
        #                    self.gap_kernel((a, a), (self.block_size_2d, self.block_size_2d), 4 * self.kn2),
        #                    self.x3, self.x2, self.size // self.stride**2, self.size // self.stride**2, self.kn2)
        # self.execute_phase("gap_y",
        #                    self.gap_kernel((a, a), (self.block_size_2d, self.block_size_2d), 4 * self.kn2),
        #                    self.y3, self.y2, self.size // self.stride ** 2, self.size // self.stride ** 2, self.kn2)

        # Dense layer;
        self.execute_phase("concat",
                           self.concat_kernel(self.num_blocks_per_processor, self.block_size_1d),
                           self.z, self.x2, self.y2, len(self.x2))
        self.execute_phase("dot_product",
                           self.dp_kernel(self.num_blocks_per_processor, self.block_size_1d),
                           self.z, self.dense_weights, self.res, len(self.z))

        # Add a final sync step to measure the real computation time;
        if self.time_phases:
            start = System.nanoTime()
        # self.gpu_result = sigmoid(self.res[0])
        self.gpu_result = self.res[0]
        # self.gpu_result = [self.x1[i] for i in range(100)]
        end = System.nanoTime()
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)

        self.benchmark.add_to_benchmark("gpu_result", self.gpu_result)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: {self.gpu_result:.4f}")
            # BenchmarkResult.log_message(
            #     f"\tgpu result: [" + ", ".join([f"{x:.2f}" for x in self.gpu_result[:100]]) + "...]")

        return self.gpu_result

    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:

        def relu(x):
            return np.maximum(x, 0)

        def conv3d2(x, kernels, shape, K, k_out, stride=1, operator=relu):
            N, M, L = shape
            out = np.zeros((N // stride) * (M // stride) * k_out)
            radius = K // 2

            for m in range(k_out):
                for i in range(0, int(np.ceil(N / stride)) - radius):
                    for j in range(0, int(np.ceil(M / stride)) - radius):
                        res = 0
                        i_f = i * stride + radius
                        j_f = j * stride + radius
                        for k_i in range(-radius, radius + 1):
                            for k_j in range(-radius, radius + 1):
                                for l in range(L):
                                    ni = i_f + k_i
                                    nj = j_f + k_j
                                    res += kernels[l + L * (k_j + radius + K * (k_i + radius + K * m))] * x[((ni * M) + nj) * L + l]
                        out[m + k_out * (j + M * i // stride)] = operator(res)
            return out

        def pooling(x, shape, K, stride):
            N, M, L = shape
            out = np.zeros((N // pooling, M // pooling, L))
            radius = K // 2
            for i in range(0, int(np.ceil(N / stride)) - radius):
                for j in range(0, int(np.ceil(M / stride)) - radius):
                    for l in range(L):
                        res = 0
                        i_f = i * stride + radius
                        j_f = j * stride + radius
                        for k_i in range(-radius, radius + 1):
                            for k_j in range(-radius, radius + 1):
                                    ni = i_f + k_i
                                    nj = j_f + k_j
                                    res += x[((ni * M) + nj) * L + l]
                        out[l + L * (j + M * i // stride)] = res / K**2
            return out

        def gap2(x, shape):
            N, M, L = shape
            out = np.zeros(L)
            for n in range(N):
                for m in range(M):
                    for i in range(L):
                        out[i] += x[i + L * (m + M * n)] / (N * M)
            return out

        def concat(x, y):
            # x and y have the same length;
            out = np.zeros(2 * len(x))
            for i in range(len(x)):
                out[i] = x[i]
                out[i + len(x)] = y[i]
            return out

        # Recompute the CPU result only if necessary;
        start = System.nanoTime()
        if self.current_iter == 0 or reinit:

            # Initialize weights;
            N = self.size
            kernel_1 = np.zeros(len(self.kernel_1))
            kernel_2 = np.zeros(len(self.kernel_2))
            kernel_3 = np.zeros(len(self.kernel_3))
            kernel_4 = np.zeros(len(self.kernel_4))
            dense_weights = np.zeros(len(self.dense_weights))
            # Random weights;
            for i in range(len(self.kernel_1)):
                kernel_1[i] = self.kernel_1[i]
                kernel_3[i] = self.kernel_3[i]
            for i in range(len(self.kernel_2)):
                kernel_2[i] = self.kernel_2[i]
                kernel_4[i] = self.kernel_4[i]

            for i in range(len(self.dense_weights)):
                dense_weights[i] = self.dense_weights[i]

            # First convolution (N,N,1) -> (N/stride,N/stride,kn1)
            x_1 = conv3d2(np.array(self.x_cpu), kernel_1, (N, N, self.channels), self.K, self.kn1, stride=self.stride)
            x_11 = pooling(x_1, (N // self.stride, N // self.stride, self.kn1), self.pooling, self.pooling)
            # Second convolution (N/stride,N/stride,kn1) -> (N/stride^2,N/stride^2,kn2)
            x_2 = conv3d2(x_11, kernel_2, (N // self.stride // self.pooling, N // self.stride // self.pooling, self.kn1), self.K, self.kn2, stride=self.stride)

            # First convolution (N,N,1) -> (N/stride,N/stride,kn1)
            y_1 = conv3d2(np.array(self.y_cpu), kernel_3, (N, N, self.channels), self.K, self.kn1, stride=self.stride)
            y_11 = pooling(y_1, (N // self.stride, N // self.stride, self.kn1), self.pooling, self.pooling)
            # Second convolution (N/stride,N/stride,kn1) -> (N/stride^2,N/stride^2,kn2)
            y_2 = conv3d2(y_11, kernel_4, (N // self.stride // self.pooling, N // self.stride // self.pooling, self.kn1), self.K, self.kn2, stride=self.stride)

            # Global average pooling 2D;
            # x_3 = gap2(x_2, (N // (self.stride * self.stride), N // (self.stride * self.stride), self.kn2))
            # y_3 = gap2(y_2, (N // (self.stride * self.stride), N // (self.stride * self.stride), self.kn2))

            # Concatenate;
            out = concat(x_2, y_2)

            # Final dense layer;
            self.cpu_result = out.dot(dense_weights[:len(out)])
            # self.cpu_result = x_1[:100]

        cpu_time = (System.nanoTime() - start) / 1_000_000_000

        # Compare GPU and CPU results;
        difference = np.abs(self.cpu_result - gpu_result)

        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", str(difference))
        if self.benchmark.debug:
            # BenchmarkResult.log_message(
            #     f"\tcpu result: [" + ", ".join([f"{x:.2f}" for x in self.cpu_result[:100]]) + "...]"+
            #                             f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")
            BenchmarkResult.log_message(f"\tcpu result: {self.cpu_result:.4f}; " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")
