# coding=utf-8
import polyglot
import time
import numpy as np
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

REDUCE_KERNEL = """
    extern "C" __global__ void reduce(float *x, float *res, int n) {
        __shared__ float cache[%d];
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            cache[threadIdx.x] = x[i];
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


class Benchmark1(Benchmark):
    """
    Compute the sum of difference of squares of 2 vectors, using multiple GrCUDA kernels.
    Structure of the computation:
       A: x^2 ──┐
                ├─> C: z=x-y ──> D: sum(z)
       B: x^2 ──┘
    """

    def __init__(self, benchmark: BenchmarkResult):
        super().__init__("b1", benchmark)
        self.size = 0
        self.x = None
        self.y = None
        self.z = None
        self.res = None
        self.num_blocks = 0
        self.square_kernel = None
        self.diff_kernel = None
        self.reduce_kernel = None
        self.cpu_result = 0

    def alloc(self, size: int):
        self.size = size
        self.num_blocks = (size + NUM_THREADS_PER_BLOCK - 1) // NUM_THREADS_PER_BLOCK

        # Allocate 2 vectors;
        start = time.time()
        self.x = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.y = polyglot.eval(language="grcuda", string=f"float[{size}]")

        # Allocate a support vector;
        self.z = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.res = polyglot.eval(language="grcuda", string=f"float[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.square_kernel = build_kernel(SQUARE_KERNEL, "square", "pointer, sint32")
        self.diff_kernel = build_kernel(DIFF_KERNEL, "diff", "pointer, pointer, pointer, sint32")
        self.reduce_kernel = build_kernel(REDUCE_KERNEL, "reduce", "pointer, pointer, sint32")

        end = time.time()
        self.benchmark.add_phase({"name": "allocation", "time_sec": end - start})

    def init(self):
        start = time.time()
        for i in range(self.size):
            self.x[i] = 1 / (i + 1)
            self.y[i] = 2 / (i + 1)
        end = time.time()
        self.benchmark.add_phase({"name": "initialization", "time_sec": end - start})

    def execute(self) -> object:
        # This must be reset at every execution;
        self.res[0] = 0

        # Call the kernel. The 2 computations are independent, and can be done in parallel;
        start_tot = time.time()
        self.square_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.x, self.size)
        self.square_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.y, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "square", "time_sec": end - start_tot})

        # C. Compute the difference of the 2 vectors. This must be done after the 2 previous computations;
        start = time.time()
        self.diff_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.x, self.y, self.z, self.size)
        end = time.time()
        self.benchmark.add_phase({"name": "diff", "time_sec": end - start})

        # D. Compute the sum of the result;
        start = time.time()
        self.reduce_kernel(self.num_blocks, NUM_THREADS_PER_BLOCK)(self.z, self.res, self.size)
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


