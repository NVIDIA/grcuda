# coding=utf-8
import polyglot
import time
import numpy as np
from random import random, randint, seed

from benchmark import Benchmark
from benchmark_result import BenchmarkResult

##############################
##############################

NUM_THREADS_PER_BLOCK = 128

SQUARE_KERNEL = """
    extern "C" __global__ void square(float* x, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            x[idx] = x[idx] * x[idx];
        }
    }
    """

DIFF_KERNEL = """
    extern "C" __global__ void diff(float* x, float* y, float* z, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            z[idx] = x[idx] - y[idx];
        }
    }
    """

ADDTWO_KERNEL = """
    extern "C" __global__ void addtwo(float* a, float* b, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            b[idx] = a[idx] + 2.0;
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


class Benchmark2(Benchmark):
    """
    Compute a complex graph of interconnected computations using GrCUDA.
    Structure of the computation:
       A: x^2 ──┐
                ├─> C: z=x-y ───┐
       B: x^2 ──┘               │
                                ├-> F: sum(z+b)
                                │
       D: a^2 ────> E: b=a+2  ──┘
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b2", benchmark, nvprof_profile)
        self.size = 0
        self.x = None
        self.y = None
        self.z = None
        self.res = None
        self.a = None
        self.b = None
        self.num_blocks = 0
        self.square_kernel = None
        self.diff_kernel = None
        self.addtwo_kernel = None
        self.reduce_kernel = None
        self.cpu_result = 0

    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.num_blocks = (size + NUM_THREADS_PER_BLOCK - 1) // NUM_THREADS_PER_BLOCK

        # Allocate 2 vectors;
        start = time.time()
        self.x = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.y = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.a = polyglot.eval(language="grcuda", string=f"float[{size}]")

        # Allocate support vectors;
        self.z = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.b = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.res = polyglot.eval(language="grcuda", string=f"float[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.square_kernel = build_kernel(SQUARE_KERNEL, "square", "pointer, sint32")
        self.diff_kernel = build_kernel(DIFF_KERNEL, "diff", "pointer, pointer, pointer, sint32")
        self.addtwo_kernel = build_kernel(ADDTWO_KERNEL, "addtwo", "pointer, pointer, sint32")
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
                self.y[i] = 2 * random()
                self.a[i] = 4 * random()
            else:
                self.x[i] = 1 / (i + 1)
                self.y[i] = 2 / (i + 1)
                self.a[i] = 4 / (i + 1)
        end = time.time()
        self.benchmark.add_phase({"name": "initialization", "time_sec": end - start})

    def execute(self) -> object:
        # This must be reset at every execution;
        self.res[0] = 0

        # Call the kernel. The 2 computations are independent, and can be done in parallel;
        start = time.time()
        self.square_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.x, self.size)
        self.square_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.y, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "square", "time_sec": end - start})

        # C. Compute the difference of the 2 vectors. This must be done after the 2 previous computations;
        start = time.time()
        self.diff_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.x, self.y, self.z, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "diff", "time_sec": end - start})

        # D. Compute the other branch of the computation;
        start = time.time()
        self.square_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.a, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "square_other_branch", "time_sec": end - start})

        # E. Continue computing the other branch;
        start = time.time()
        self.addtwo_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.a, self.b, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "add_two_other_branch", "time_sec": end - start})

        # F. Compute the sum of the result;
        start = time.time()
        self.reduce_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.z, self.b, self.res, self.size)
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
                a_g = np.zeros(self.size)
                for i in range(self.size):
                    x_g[i] = random()
                    y_g[i] = 2 * random()
                    a_g[i] = 4 * random()
            else:
                x_g = 1 / np.linspace(1, self.size, self.size)
                y_g = 2 / np.linspace(1, self.size, self.size)
                a_g = 4 / np.linspace(1, self.size, self.size)

            x_g = x_g ** 2
            y_g = y_g ** 2
            a_g = a_g ** 2
            x_g -= y_g
            a_g += 2
            self.cpu_result = np.sum(x_g + a_g)
        cpu_time = time.time() - start
        difference = np.abs(self.cpu_result - gpu_result)
        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", difference)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: {self.cpu_result:.4f}, " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


