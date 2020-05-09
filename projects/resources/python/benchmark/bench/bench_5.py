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

SQUARE_KERNEL = """
    extern "C" __global__ void square(const float* x, float* y, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            y[idx] = x[idx] * x[idx];
        }
    }
    """

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


class Benchmark5(Benchmark):
    """
    Compute a complex pipeline of kernels, doing mock computations, and using read-only arguments;
                         ┌─> B(const Y, R1) ───────────────────┐
        A: (const X, Y) ─┤                                     ├─> E(const R1, const R2, R)
                         └─> C(const Y, Z) ─> D(const Z, R2) ──┘
    """

    def __init__(self, benchmark: BenchmarkResult):
        super().__init__("b5", benchmark)
        self.size = 0
        self.x = None
        self.y = None
        self.z = None
        self.r1 = None
        self.r2 = None
        self.r = None
        self.num_blocks = 0
        self.kernel_a = None
        self.kernel_b = None
        self.kernel_c = None
        self.kernel_d = None
        self.kernel_e = None
        self.cpu_result = None

    @time_phase("allocation")
    def alloc(self, size: int):
        self.size = size
        self.num_blocks = (size + NUM_THREADS_PER_BLOCK - 1) // NUM_THREADS_PER_BLOCK

        # Allocate vectors;
        self.x = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.y = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.z = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.r1 = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.r2 = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.r = polyglot.eval(language="grcuda", string=f"float[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.kernel_a = build_kernel(SQUARE_KERNEL, "square", "const pointer, pointer, sint32")
        self.kernel_b = build_kernel(COMPUTE_KERNEL, "compute", "const pointer, pointer, sint32")
        self.kernel_c = build_kernel(SQUARE_KERNEL, "square", "const pointer, pointer, sint32")
        self.kernel_d = build_kernel(COMPUTE_KERNEL, "compute", "const pointer, pointer, sint32")
        self.kernel_e = build_kernel(REDUCE_KERNEL, "reduce", "const pointer, const pointer, pointer, sint32")

    @time_phase("initialization")
    def init(self):
        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)
        for i in range(self.size):
            if self.benchmark.random_init:
                self.x[i] = random()
            else:
                self.x[i] = 1 / (i + 1)

    def execute(self) -> object:
        # This must be reset at every execution;
        self.r[0] = 0

        # A.
        start = time.time()
        self.kernel_a(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.x, self.y, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "kernel_a", "time_sec": end - start})

        # B.
        start = time.time()
        self.kernel_b(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.y, self.r1, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "kernel_b", "time_sec": end - start})

        # C, D.
        start = time.time()
        self.kernel_c(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.y, self.z, self.size)
        self.kernel_d(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.z, self.r2, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "kernel_c_d", "time_sec": end - start})

        # E.
        start = time.time()
        self.kernel_e(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.r1, self.r2, self.r, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "kernel_e", "time_sec": end - start})

        # Read the result;
        start = time.time()
        result = self.r[0]
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
                for i in range(self.size):
                    x_g[i] = random()
            else:
                x_g = 1 / np.linspace(1, self.size, self.size)

            x_g = x_g**2
            r1 = np.log(1 + np.sin(x_g)**2 + np.cos(x_g)**2)
            r2 = np.log(1 + np.sin(x_g**2) ** 2 + np.cos(x_g**2) ** 2)
            self.cpu_result = np.sum(r1 + r2)

        cpu_time = time.time() - start
        difference = np.abs(self.cpu_result - gpu_result)
        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", difference)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: {self.cpu_result:.4f}, " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


