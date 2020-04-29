# coding=utf-8
import polyglot
import time
import numpy as np
from random import random, randint, seed

from benchmark import Benchmark, time_phase
from benchmark_result import BenchmarkResult

##############################
##############################

NUM_THREADS_PER_BLOCK = 128

COMPUTE_KERNEL = """
    extern "C" __global__ void compute(const float* x, float *y, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            y[i] = logf(1.0 + cosf(x[i]) * cosf(x[i]) + sinf(x[i]) * sinf(x[i]));
        }
    }
    """

REDUCE_KERNEL = """
    extern "C" __global__ void reduce(const float *x, const float *y, float *res, int n) {
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


class Benchmark4(Benchmark):
    """
    Compute the sum of 2 vectors after having applied some computationally intensive arithmetic to it:
       A: log(1 + sin(x)^2 + cos(x)^2) ──┐
                                         ├─> C: sum(x + y)
       B: log(1 + sin(y)^2 + cos(y)^2) ──┘
    """

    def __init__(self, benchmark: BenchmarkResult):
        super().__init__("b4", benchmark)
        self.size = 0
        self.x = None
        self.y = None
        self.a = None
        self.b = None
        self.res = None
        self.num_blocks = 0
        self.compute_kernel = None
        self.reduce_kernel = None
        self.cpu_result = 0

    @time_phase("allocation")
    def alloc(self, size: int):
        self.size = size
        self.num_blocks = (size + NUM_THREADS_PER_BLOCK - 1) // NUM_THREADS_PER_BLOCK

        # Allocate 4 vectors;
        self.x = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.y = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.a = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.b = polyglot.eval(language="grcuda", string=f"float[{size}]")

        # Allocate a support vector;
        self.res = polyglot.eval(language="grcuda", string=f"float[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.compute_kernel = build_kernel(COMPUTE_KERNEL, "compute", "pointer, pointer, sint32")
        self.reduce_kernel = build_kernel(REDUCE_KERNEL, "reduce", "pointer, pointer, pointer, sint32")

    @time_phase("initialization")
    def init(self):
        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)
        for i in range(self.size):
            if self.benchmark.random_init:
                self.x[i] = random()
                self.a[i] = random()
            else:
                self.x[i] = 1 / (i + 1)
                self.a[i] = 1 / (i + 1)

    def execute(self) -> object:
        # This must be reset at every execution;
        self.res[0] = 0

        # A. B. Call the kernels. The 2 computations are independent, and can be done in parallel;
        start = time.time()
        self.compute_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.x, self.y, self.size)
        self.compute_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.a, self.b, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "compute", "time_sec": end - start})

        # C. Compute the sum of the result;
        start = time.time()
        self.reduce_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.y, self.b, self.res, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "reduce", "time_sec": end - start})

        start = time.time()
        result = self.res[0]
        end = time.time()
        self.benchmark.add_phase({"name": "read_result", "time_sec": end - start})

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

            x_g = np.log(1 + np.sin(x_g)**2 + np.cos(x_g)**2)
            y_g = np.log(1 + np.sin(y_g)**2 + np.cos(y_g)**2)
            self.cpu_result = np.sum(x_g + y_g)
        cpu_time = time.time() - start
        difference = np.abs(self.cpu_result - gpu_result)
        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", difference)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: {self.cpu_result:.4f}, " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


