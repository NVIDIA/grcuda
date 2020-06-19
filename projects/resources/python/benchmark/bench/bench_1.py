# coding=utf-8
import polyglot
import time
import numpy as np
from random import random, randint, seed

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D
from benchmark_result import BenchmarkResult

##############################
##############################

SQUARE_KERNEL = """
extern "C" __global__ void square(const float* x, float* y, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] * x[i];
    }
}
"""

DIFF_KERNEL = """
extern "C" __global__ void diff(const float* x, const float* y, float* z, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        z[i] = x[i] - y[i];
    }
}
"""

REDUCE_KERNEL = """

// From https://devblogs.nvidia.com/faster-parallel-reductions-kepler/

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduce(const float *x, const float *y, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] - y[i];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}
"""

##############################
##############################


class Benchmark1(Benchmark):
    """
    Compute the sum of difference of squares of 2 vectors, using multiple GrCUDA kernels.
    Structure of the computation:
       A: x^2 ──┐
                ├─> C: z=sum(x-y)
       B: x^2 ──┘
    """

    def __init__(self, benchmark: BenchmarkResult):
        super().__init__("b1", benchmark)
        self.size = 0
        self.x = None
        self.y = None
        self.x1 = None
        self.y1 = None
        self.z = None
        self.res = None
        self.square_kernel = None
        self.diff_kernel = None
        self.reduce_kernel = None
        self.cpu_result = 0

        self.num_blocks = 64
        self.block_size = DEFAULT_BLOCK_SIZE_1D

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.block_size = block_size["block_size_1d"]

        # Allocate 2 vectors;
        self.x = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.y = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.x1 = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.y1 = polyglot.eval(language="grcuda", string=f"float[{size}]")

        # Allocate a support vector;
        self.res = polyglot.eval(language="grcuda", string=f"float[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.square_kernel = build_kernel(SQUARE_KERNEL, "square", "const pointer, pointer, sint32")
        self.reduce_kernel = build_kernel(REDUCE_KERNEL, "reduce", "const pointer, const pointer, pointer, sint32")

    @time_phase("initialization")
    def init(self):
        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)
        for i in range(self.size):
            if self.benchmark.random_init:
                self.x[i] = random()
                self.y[i] = 2 * random()
            else:
                self.x[i] = 1 / (i + 1)
                self.y[i] = 2 / (i + 1)

    @time_phase("reset_result")
    def reset_result(self) -> None:
        self.res[0] = 0.0

    def execute(self) -> object:

        # A, B. Call the kernel. The 2 computations are independent, and can be done in parallel;
        start = time.time()
        self.square_kernel(self.num_blocks, self.block_size)(self.x, self.x1, self.size)
        self.square_kernel(self.num_blocks, self.block_size)(self.y, self.y1, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "square", "time_sec": end - start})

        # C. Compute the sum of the result;
        start = time.time()
        self.reduce_kernel(self.num_blocks, self.block_size)(self.x1, self.y1, self.res, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "reduce", "time_sec": end - start})

        # Add a final sync step to measure the real computation time;
        start = time.time()
        result = self.res[0]
        end = time.time()
        self.benchmark.add_phase({"name": "sync", "time_sec": end - start})

        self.benchmark.add_to_benchmark("gpu_result", result)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: {result:.4f}")

        return result

    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:
        # Recompute the CPU result only if necessary;
        start = time.time()
        if self.current_iter == 0 or reinit:
            # Re-initialize the random number generator with the same seed as the GPU to generate the same values;
            seed(self.random_seed)
            if self.benchmark.random_init:
                x_g = np.zeros(self.size)
                y_g = np.zeros(self.size)
                for i in range(self.size):
                    x_g[i] = random()
                    y_g[i] = 2 * random()
            else:
                x_g = 1 / np.linspace(1, self.size, self.size)
                y_g = 2 / np.linspace(1, self.size, self.size)

            x_g = x_g ** 2
            y_g = y_g ** 2
            x_g -= y_g
            self.cpu_result = np.sum(x_g)
        cpu_time = time.time() - start
        difference = np.abs(self.cpu_result - gpu_result)
        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", difference)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: {self.cpu_result:.4f}, " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


