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
import time
import numpy as np
from random import random, randint, seed

from benchmark import Benchmark
from benchmark_result import BenchmarkResult

##############################
##############################

NUM_ITER = 5

NUM_THREADS_PER_BLOCK = 128

SQUARE_KERNEL = """
    extern "C" __global__ void square(float* x, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            x[idx] = x[idx] * x[idx];
        }
    }
    """

REDUCE_KERNEL = """
    extern "C" __global__ void reduce(float *x, float *y, float *res, int n) {
        __shared__ float cache[%d];
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            cache[threadIdx.x] = x[i] + y[i];
        }
        __syncthreads();

        // Perform tree reduction;
        i = %d / 2;
        while (i > 0) {
            if (threadIdx.x < i) {
                cache[threadIdx.x] += cache[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }
        if (threadIdx.x == 0) {
            atomicAdd(res, cache[0]);
        }
    }
    """ % (NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK)

##############################
##############################


class Benchmark3(Benchmark):
    """
    Compute a pipeline of GrCUDA kernels using loops to build a dynamic graph.
    Structure of the computation:
       A: x^2 ─ [5 times] ─┐
                           ├─> C: res=sum(x+y)
       B: x^2 ─ [5 times] ─┘
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b3", benchmark, nvprof_profile)
        self.size = 0
        self.x = None
        self.y = None
        self.res = None
        self.num_blocks = 0
        self.square_kernel = None
        self.reduce_kernel = None
        self.cpu_result = 0
        self.num_iter = NUM_ITER

    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.num_blocks = (size + NUM_THREADS_PER_BLOCK - 1) // NUM_THREADS_PER_BLOCK

        # Allocate 2 vectors;
        start = time.time()
        self.x = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.y = polyglot.eval(language="grcuda", string=f"float[{size}]")

        # Allocate a support vector;
        self.res = polyglot.eval(language="grcuda", string=f"float[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.square_kernel = build_kernel(SQUARE_KERNEL, "square", "pointer, sint32")
        self.reduce_kernel = build_kernel(REDUCE_KERNEL, "reduce", "pointer, pointer, pointer, sint32")

        end = time.time()
        self.benchmark.add_phase({"name": "allocation", "time_sec": end - start})

    def init(self):
        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)
        start = time.time()
        for i in range(self.size):
            if self.benchmark.random_init:
                self.x[i] = random()
                self.y[i] = random()
            else:
                self.x[i] = 1 / (i + 1)
                self.y[i] = 1 / (i + 1)
        end = time.time()
        self.benchmark.add_phase({"name": "initialization", "time_sec": end - start})

    def execute(self) -> object:
        # This must be reset at every execution;
        self.res[0] = 0

        # A. B. Call the kernels. The 2 computations are independent, and can be done in parallel;
        for i in range(self.num_iter):
            start = time.time()
            self.square_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.x, self.size)
            self.square_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.y, self.size)
            end = time.time()
            self.benchmark.add_phase({"name": f"square_{i}", "time_sec": end - start})

        # C. Compute the sum of the result;
        start = time.time()
        self.reduce_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.x, self.y, self.res, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "reduce", "time_sec": end - start})

        result = self.res[0]
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
                    y_g[i] = random()
            else:
                x_g = 1 / np.linspace(1, self.size, self.size)
                y_g = 1 / np.linspace(1, self.size, self.size)

            for i in range(NUM_ITER):
                x_g = x_g ** 2
                y_g = y_g ** 2
            self.cpu_result = np.sum(x_g + y_g)
        cpu_time = time.time() - start
        difference = np.abs(self.cpu_result - gpu_result)
        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", difference)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: {self.cpu_result:.4f}, " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


