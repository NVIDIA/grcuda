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

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D
from benchmark_result import BenchmarkResult
from java.lang import System

##############################
##############################

SUM_KERNEL = """
extern "C" __global__ void sum(int* x, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] += 1;
    }
}
"""

##############################
##############################


class Benchmark4(Benchmark):
    """
    A benchmark with 2 very simple independent computations, used to measure overheads and the impact of data transfer;
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b4", benchmark, nvprof_profile)
        self.size = 0
        self.x = None
        self.y = None
        self.num_blocks = 64
        self.sum_kernel = None
        self.cpu_result = 0
        self.block_size = DEFAULT_BLOCK_SIZE_1D

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.block_size = block_size["block_size_1d"]

        # Allocate 4 vectors;
        self.x = polyglot.eval(language="grcuda", string=f"int[{size}]")
        self.y = polyglot.eval(language="grcuda", string=f"int[{size}]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.sum_kernel = build_kernel(SUM_KERNEL, "sum", "pointer, sint32")

    @time_phase("initialization")
    def init(self):
        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)
        for i in range(self.size):
            if self.benchmark.random_init:
                self.x[i] = randint(0, 10)
                self.y[i] = randint(0, 10)
            else:
                self.x[i] = 1 / (i + 1)
                self.y[i] = 1 / (i + 1)

    def execute(self) -> object:

        # A. B. Call the kernels. The 2 computations are independent, and can be done in parallel;
        start = System.nanoTime()
        self.sum_kernel(self.num_blocks, self.block_size)(self.x, self.size)
        end = System.nanoTime()
        self.benchmark.add_phase({"name": "sum_1", "time_sec": (end - start) / 1_000_000_000})

        start = System.nanoTime()
        self.sum_kernel(self.num_blocks, self.block_size)(self.y, self.size)
        end = System.nanoTime()
        self.benchmark.add_phase({"name": "sum_2", "time_sec": (end - start) / 1_000_000_000})

        start = System.nanoTime()
        result_1 = self.x[0]
        result_2 = self.y[0]
        end = System.nanoTime()
        self.benchmark.add_phase({"name": "read_result", "time_sec": (end - start) / 1_000_000_000})

        self.benchmark.add_to_benchmark("gpu_result", result_1 + result_2)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: {result_1} {result_2}")

        return result_1 + result_2

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
                    x_g[i] = randint(0, 10)
                    y_g[i] = randint(0, 10)
            else:
                x_g = 1 / np.linspace(1, self.size, self.size)
                y_g = 1 / np.linspace(1, self.size, self.size)

            x_g += 1
            y_g += 1
            self.cpu_result = x_g[0] + y_g[0]
        cpu_time = System.nanoTime() - start
        difference = np.abs(self.cpu_result - gpu_result)
        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", difference)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: {self.cpu_result:.4f}, " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


