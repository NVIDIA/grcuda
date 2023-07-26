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
from random import random, randint, seed

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D, DEFAULT_NUM_BLOCKS
from benchmark_result import BenchmarkResult

##############################
##############################

SQUARE_KERNEL = """
extern "C" __global__ void square(float* x, float* y, int n) {
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

__global__ void reduce(float *x, float *y, float* z, int N) {
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
    It's a fairly artificial benchmark that measures a simple case of parallelism.
    Most of the execution time is spent in the reduction computation, limiting the amount of parallelism available, 
    especially on large input data.
    Speedups are achievable by overlapping data-transfer and computations, 
    although the data-transfer takes about 4x-5x longer than the square computation, limiting the maximum achievable speedup.

    Structure of the computation:

    A: x^2 ──┐
            ├─> C: z=sum(x-y)
    B: x^2 ──┘
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b1", benchmark, nvprof_profile)
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

        # self.num_blocks = DEFAULT_NUM_BLOCKS
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
        self.square_kernel = build_kernel(SQUARE_KERNEL, "square", "pointer, pointer, sint32")
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
        if self.benchmark.random_init:
            seed(self.random_seed)
            for i in range(self.size):
                self.x[i] = random()
                self.y[i] = 2 * random()
        else:
            for i in range(self.size):
                self.x[i] = 1 / (i + 1)
                self.y[i] = 2 / (i + 1)
        self.res[0] = 0.0

    def execute(self) -> object:
        self.block_size = self._block_size["block_size_1d"]
        start_comp = System.nanoTime()
        start = 0

        # A, B. Call the kernel. The 2 computations are independent, and can be done in parallel;
        self.execute_phase("square_1", self.square_kernel(self.num_blocks, self.block_size), self.x, self.x1, self.size)
        self.execute_phase("square_2", self.square_kernel(self.num_blocks, self.block_size), self.y, self.y1, self.size)

        # C. Compute the sum of the result;
        self.execute_phase("reduce", self.reduce_kernel(self.num_blocks, self.block_size), self.x1, self.y1, self.res, self.size)

        # Add a final sync step to measure the real computation time;
        if self.time_phases:
            start = System.nanoTime()
        result = self.res[0]
        end = System.nanoTime()
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)
        self.benchmark.add_to_benchmark("gpu_result", result)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: {result:.4f}")

        return result

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


