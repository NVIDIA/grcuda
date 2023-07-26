# coding=utf-8
import polyglot
from java.lang import System
import numpy as np
from random import random, randint, seed, sample

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D
from benchmark_result import BenchmarkResult

##############################
##############################

BLOCK_SIZE_V100 = 64  # Just a recommendation of optimal block size for the V100;
P = 16
ITER = 50

PRECONDITION_KERNEL = """
// Add a small epsilon to the main diagonal:
extern "C" __global__ void precondition(float *A, int n, int m, int offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
        A[i * n + i + offset] += 1e-12; 
    }
}
"""

MMUL_KERNEL = """
// z = x @ y;
extern "C" __global__ void matrix_vector_mult(const float* x, const float* y, float* z, int n, int m, int z_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[z_offset + i] = sum;
    }
}

// z := w + alpha * A @ y;
extern "C" __global__ void matrix_vector_mult_axpy(const float* x, const float* y, const float *w, const float alpha, float* z, int n, int m, int z_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[z_offset + i] = alpha * sum + w[z_offset + i];
    }
}
"""

DP_KERNEL = """
__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// z = <x, x>;
extern "C" __global__ void l2_norm(const float *x, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float x_tmp = x[i];
        sum += x_tmp * x_tmp;
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

// z = <x, y>;
extern "C" __global__ void dot(const float *x, const float *y, float* z, int N) {
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

SAXPY_KERNEL = """
// y = val + alpha * x;
extern "C" __global__ void saxpy(float* y, const float *val, const float *x, float alpha, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = val[i] + alpha * x[i];
    }
}

// Simply copy array x into y;
extern "C" __global__ void cpy(float *y, const float *x, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i];
    }
}
"""

##############################
##############################


class Benchmark9M(Benchmark):
    """
    Compute the conjugate gradient algorithm on a dense symmetric matrix.
    The matrix-vector multiplications are row-partitioned to scale across multiple GPUs;
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b9m", benchmark, nvprof_profile)
        self.size = 0
        self.S = 0
        self.A = [None for _ in range(P)]
        self.x = None
        self.b = None
        self.p = None
        self.r = None
        self.y = None
        self.t1 = None
        self.t2 = None

        self.num_blocks_size = BLOCK_SIZE_V100
        self.block_size = None

        self.mmul_axpy_kernel = None
        self.mmul_kernel = None
        self.l2_norm_kernel = None
        self.dp_kernel = None
        self.saxpy_kernel = None
        self.copy_kernel = None

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.S = (self.size + P - 1) // P
        self.block_size = self._block_size["block_size_1d"]

        self.random_seed = 12
        seed(self.random_seed)

        # Allocate vectors;
        for i in range(P):
            self.A[i] = polyglot.eval(language="grcuda", string=f"float[{self.size * self.S}]")
        self.x = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.b = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.p = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.r = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.y = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.t1 = polyglot.eval(language="grcuda", string=f"float[1]")
        self.t2 = polyglot.eval(language="grcuda", string=f"float[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.precondition_kernel = build_kernel(PRECONDITION_KERNEL, "precondition", "pointer, sint32, sint32, sint32")
        self.mmul_kernel = build_kernel(MMUL_KERNEL, "matrix_vector_mult", "const pointer, const pointer, const pointer, sint32, sint32, sint32")
        self.mmul_axpy_kernel = build_kernel(MMUL_KERNEL, "matrix_vector_mult_axpy", "const pointer, const pointer, const pointer, float, const pointer, sint32, sint32, sint32")
        self.l2_norm_kernel = build_kernel(DP_KERNEL, "l2_norm", "const pointer, pointer, sint32")
        self.dp_kernel = build_kernel(DP_KERNEL, "dot", "const pointer, pointer, pointer, sint32")
        self.saxpy_kernel = build_kernel(SAXPY_KERNEL, "saxpy", "pointer, const pointer, const pointer, float, sint32")
        self.cpy_kernel = build_kernel(SAXPY_KERNEL, "cpy", "pointer, pointer, sint32")
        self.initialize_random_symmetric_matrix = polyglot.eval(language="js", string="""(X, S, N) => { 
            for (let i = 0; i < N; i++) {
                s = (i / S) >> 0;
                k = i % S;
                Xs = X[s];
                i_N = k * N;
                for (let j = i; j < N; j++) {
                    val = 2 * Math.random() - 1; 
                    Xs[i_N + j] = val;
                    X[(j / S) >> 0][(j % S) * N + i] = val;
                }
            }}
            """)

    @time_phase("initialization")
    def init(self):
        self.initialize_random_symmetric_matrix(self.A, self.S, self.size)

    @time_phase("reset_result")
    def reset_result(self) -> None:
        seed(self.random_seed)
        # Random initial solution;
        for i in range(self.size):
            self.x[i] = 1.0 / self.size
        self.t1[0] = 0.0
        self.t2[0] = 0.0

    def execute(self) -> object:
      
        start_comp = System.nanoTime()
        start = 0
        
        # Initialization phase;
        # precondition: A += I * np.eps;
        for i in range(P):
            self.execute_phase(f"precondition_{i}", self.precondition_kernel(self.num_blocks, self.block_size),
                               self.A[i], self.size, min(self.S, self.size - i * self.S), i * self.S)
        # r = b - A * x
        for i in range(P):
            self.execute_phase(f"mmul_init_{i}", self.mmul_axpy_kernel(self.num_blocks, self.block_size),
                               self.A[i], self.x, self.b, -1, self.r, self.S, self.size, i * self.S)
        # p = r
        self.execute_phase("cpy_init", self.cpy_kernel(self.num_blocks, self.block_size),
                           self.p, self.r, self.size)
        # t1 = r^t * r
        self.execute_phase("norm_init", self.l2_norm_kernel(self.num_blocks, self.block_size),
                           self.r, self.t1, self.size)
        for curr_iter in range(ITER):
            # t2 = p^t * A * p
            for i in range(P):
                self.execute_phase(f"mmul_{i}_{curr_iter}", self.mmul_kernel(self.num_blocks, self.block_size),
                                self.A[i], self.p, self.y, self.S, self.size, i * self.S)
            self.execute_phase(f"dp_{curr_iter}", self.dp_kernel(self.num_blocks, self.block_size), 
                               self.p, self.y, self.t2, self.size)
            
            if self.time_phases:
                start = System.nanoTime()
            alpha = self.t1[0] / self.t2[0]
            old_r_norm_squared = self.t1[0]
            self.t1[0] = 0
            self.t2[0] = 0
            if self.time_phases:
                end = System.nanoTime()
                self.benchmark.add_phase({"name": f"alpha_{curr_iter}", "time_sec": (end - start) / 1_000_000_000})

            # Update x: x = x + alpha * p
            self.execute_phase(f"saxpy_x_{curr_iter}", self.saxpy_kernel(self.num_blocks, self.block_size),
                               self.x, self.x, self.p, alpha, self.size)
            # r = r - alpha * y
            self.execute_phase(f"saxpy_r_{curr_iter}", self.saxpy_kernel(self.num_blocks, self.block_size),
                               self.r, self.r, self.y, -1 * alpha, self.size)
            # t1 = r^t * r
            self.execute_phase(f"norm_{curr_iter}", self.l2_norm_kernel(self.num_blocks, self.block_size), 
                               self.r, self.t1, self.size)

            if self.time_phases:
                start = System.nanoTime()
            beta = self.t1[0] / old_r_norm_squared
            if self.time_phases:
                end = System.nanoTime()
                self.benchmark.add_phase({"name": f"beta_{curr_iter}", "time_sec": (end - start) / 1_000_000_000})

            self.execute_phase(f"saxpy_p_{curr_iter}", self.saxpy_kernel(self.num_blocks, self.block_size),
                               self.p, self.r, self.p, beta, self.size)

        # Add a final sync step to measure the real computation time;
        if self.time_phases:
            start = System.nanoTime()
        tmp = self.x[0]
        end = System.nanoTime()
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)
        # Compute GPU result;
        for i in range(P):
            self.mmul_axpy_kernel(self.num_blocks, self.block_size)(self.A[i], self.x, self.b, -1, self.y, min(self.S, self.size - i * self.S), self.size, i * self.S)

        self.gpu_result = sum(self.y[:10])
        self.benchmark.add_to_benchmark("gpu_result", 0)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: [" + ", ".join([f"{x:.4f}" for x in self.y[:10]]) + f"...] = {self.gpu_result:.4f}")

        return self.gpu_result

    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:

        # Recompute the CPU result only if necessary;
        start = System.nanoTime()
        x_cpu = np.zeros(self.size)
        A_cpu = np.zeros((self.size, self.size))
        if self.current_iter == 0 or reinit:
            # Re-initialize the random number generator with the same seed as the GPU to generate the same values;
            seed(self.random_seed)
            # Initialize the support device arrays;
            N = self.size

            for i in range(N):
                p = i // self.S
                for j in range(N):
                    A_cpu[i, j] = self.A[p][(i % self.S) * N + j]

            b = np.random.random(N)
            x = np.ones(N)
            r = b - A_cpu @ x
            p = r.copy()
            t1 = r.T.dot(r)

            # Main iteration;
            for i in range(ITER):
                y = A_cpu @ p
                t2 = p.dot(y)
                alpha = t1 / t2
                t1_old = t1
                x += alpha * p
                r -= alpha * y
                t1 = r.T.dot(r)
                beta = t1 / t1_old
                p = r + beta * p

            self.cpu_result = x

        cpu_time = System.nanoTime() - start

        # Compare GPU and CPU results;
        difference = np.abs(self.cpu_result - gpu_result)

        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", str(difference))
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: [" + ", ".join([f"{x:.4f}" for x in x_cpu[:10]]) + "...]; " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")




