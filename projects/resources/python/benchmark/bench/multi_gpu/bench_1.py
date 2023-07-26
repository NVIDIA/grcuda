# coding=utf-8
import polyglot
from java.lang import System
import numpy as np
from random import random, seed

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D, DEFAULT_NUM_BLOCKS
from benchmark_result import BenchmarkResult

##############################
##############################

# Number of partitions;
P = 16

SQUARE_KERNEL = """
extern "C" __global__ void square(const float *x, float *y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] * x[i];  
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

extern "C" __global__ void reduce(const float *x, const float *y, float *z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] - y[i];
    }
    sum = warp_reduce(sum);                    // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum);                     // The first thread in the warp updates the output;
}
"""

##############################
##############################


class Benchmark1M(Benchmark):
    """
    Compute the sum of difference of squares of 2 vectors, using multiple GrCUDA kernels.
    Parallelize the computation on multiple GPUs, by computing a chunk of the output on each.
    Then aggregate results on the CPU;
    Structure of the computation:
    * GPU0:
       A: x^2 ──┐
                ├─> C: z0=sum(x-y)
       B: x^2 ──┘
    * GPU1:
       A: x^2 ──┐
                ├─> C: z1=sum(x-y)
       B: x^2 ──┘
    * GPU2: [...]
    * CPU: z = z0 + z1 + ...
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b1m", benchmark, nvprof_profile)
        self.size = 0
        self.S = 0
        self.x = None
        self.y = None
        self.x1 = None
        self.y1 = None
        self.z = None
        self.res = None
        self.square_kernel = None
        self.diff_kernel = None
        self.reduce_kernel = None
        self.initialize = None
        self.res_tot = 0
        self.cpu_result = 0

        # self.num_blocks = DEFAULT_NUM_BLOCKS
        self.block_size = DEFAULT_BLOCK_SIZE_1D

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.block_size = block_size["block_size_1d"]

        # Number of items in each partition;
        self.S = (self.size + P - 1) // P

        self.x = [None for _ in range(P)]
        self.y = [None for _ in range(P)]
        self.x1 = [None for _ in range(P)]
        self.y1 = [None for _ in range(P)]
        self.res = [None for _ in range(P)]

        # Allocate 2 vectors;
        for i in range(P):
            self.x[i] = polyglot.eval(language="grcuda", string=f"float[{self.S}]")
            self.y[i] = polyglot.eval(language="grcuda", string=f"float[{self.S}]")
            self.x1[i] = polyglot.eval(language="grcuda", string=f"float[{self.S}]")
            self.y1[i] = polyglot.eval(language="grcuda", string=f"float[{self.S}]")
            self.res[i] = polyglot.eval(language="grcuda", string=f"float[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.square_kernel = build_kernel(SQUARE_KERNEL, "square", "const pointer, pointer, sint32")
        self.reduce_kernel = build_kernel(REDUCE_KERNEL, "reduce", "const pointer, const pointer, pointer, sint32")

        self.initialize = polyglot.eval(language="js", string="(x, i, N, a) => { for (let j = 0; j < x.length; j++) { let index = i * x.length + j; if (index < N) {x[j] = a / (index + 1); }}}")


    @time_phase("initialization")
    def init(self):
        for i in range(P):
            self.initialize(self.x[i], i, self.size, 1)
            self.initialize(self.y[i], i, self.size, 2)

    @time_phase("reset_result")
    def reset_result(self) -> None:
        for i in range(P):
            self.initialize(self.x[i], i, self.size, 1)
            self.initialize(self.y[i], i, self.size, 2)
            self.res[i][0] = 0.0
        self.res_tot = 0

    def execute(self) -> object:
        self.block_size = self._block_size["block_size_1d"]
        start_comp = System.nanoTime()
        start = 0
        for i in range(P):
            # A, B. Call the kernel. The 2 computations are independent, and can be done in parallel;
            self.execute_phase(f"square_1_{i}", self.square_kernel(self.num_blocks, self.block_size), self.x[i], self.x1[i], self.S)
            self.execute_phase(f"square_2_{i}", self.square_kernel(self.num_blocks, self.block_size), self.y[i], self.y1[i], self.S)
            # C. Compute the sum of the result;
            self.execute_phase(f"reduce_{i}", self.reduce_kernel(self.num_blocks, self.block_size), self.x1[i], self.y1[i], self.res[i], self.S)

        # Add a final sync step to measure the real computation time;
        if self.time_phases:
            start = System.nanoTime()
        for i in range(P):
            self.res_tot += self.res[i][0]
        end = System.nanoTime()
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)
        self.benchmark.add_to_benchmark("gpu_result", self.res_tot)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: {self.res_tot:.4f}")

        return self.res_tot

    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:
        # Recompute the CPU result only if necessary;
        start = System.nanoTime()
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
        cpu_time = System.nanoTime() - start
        difference = np.abs(self.cpu_result - gpu_result)
        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", difference)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: {self.cpu_result:.4f}, " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


