# coding=utf-8
import polyglot
import time
import numpy as np
from random import random, randint, seed

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D
from benchmark_result import BenchmarkResult
from java.lang import System
import math

##############################
##############################

R = 0.08
V = 0.3
T = 1.0
K = 60.0

BS_KERNEL = """
__device__ inline float cndGPU(float d) {
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

extern "C" __global__ void bs(const float *x, float *y, int N, float R, float V, float T, float K) {

    float sqrtT = __fdividef(1.0F, rsqrtf(T));
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float expRT;
        float d1, d2, CNDD1, CNDD2;
        d1 = __fdividef(__logf(x[i] / K) + (R + 0.5f * V * V) * T, V * sqrtT);
        d2 = d1 - V * sqrtT;

        CNDD1 = cndGPU(d1);
        CNDD2 = cndGPU(d2);

        //Calculate Call and Put simultaneously
        expRT = __expf(-R * T);
        y[i] = x[i] * CNDD1 - K * expRT * CNDD2;
    }
}
"""

##############################
##############################


class Benchmark5(Benchmark):
    """
    Black & Scholes equation benchmark, executed concurrently on different input vectors;
    """

    def __init__(self, benchmark: BenchmarkResult):
        super().__init__("b5", benchmark)
        self.size = 0

        self.num_blocks = 64
        self.sum_kernel = None
        self.cpu_result = 0
        self.block_size = DEFAULT_BLOCK_SIZE_1D

        self.K = 10
        self.x = [[]] * self.K
        self.x_tmp = None
        self.y = [[]] * self.K

        self.bs_kernel = None

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.block_size = block_size["block_size_1d"]
        self.x_tmp = None
        # self.x_tmp = [0] * self.size

        # Allocate vectors;
        for i in range(self.K):
            self.x[i] = polyglot.eval(language="grcuda", string=f"float[{size}]")
            self.y[i] = polyglot.eval(language="grcuda", string=f"float[{size}]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.bs_kernel = build_kernel(BS_KERNEL, "bs", "const pointer, pointer, sint32, float, float, float, float")

    @time_phase("initialization")
    def init(self):
        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)
        if self.benchmark.random_init:
            self.x_tmp = np.random.uniform(-0.5, 0.5, self.size).astype(np.float32) + K
        else:
            self.x_tmp = np.zeros(self.size, dtype=np.float32) + K

        # for i in range(self.size):
        #     if self.benchmark.random_init:
        #         self.x_tmp[i] = random() + K - 0.5
        #     else:
        #         self.x_tmp[i] = K

    @time_phase("reset_result")
    def reset_result(self) -> None:
        for i in range(self.K):
            self.x[i].copyFrom(int(np.int64(self.x_tmp.ctypes.data)), self.size)
            # for j in range(self.size):
            #     self.x[i][j] = self.x_tmp[j]

    def execute(self) -> object:

        result = [0] * self.K

        # Call the kernels;
        start_comp = System.nanoTime()
        start = System.nanoTime()
        for i in range(self.K):
            self.execute_phase(f"bs_{i}", self.bs_kernel(self.num_blocks, self.block_size), self.x[i], self.y[i], self.size, R, V, T, K)

        if self.time_phases:
            start = System.nanoTime()
        for i in range(self.K):
            result[i] = self.y[i][0]
        end = System.nanoTime()
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)

        self.benchmark.add_to_benchmark("gpu_result", result[0])
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: {result[0]}")

        return result[0]

    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:

        def CND(X):
            """
            Cumulative normal distribution.
            Helper function used by BS(...).
            """

            (a1, a2, a3, a4, a5) = (0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
            L = np.absolute(X)
            K = np.float64(1.0) / (1.0 + 0.2316419 * L)
            w = 1.0 - 1.0 / math.sqrt(2 * np.pi) * np.exp(-L * L / 2.) * \
                (a1 * K +
                 a2 * (K ** 2) +
                 a3 * (K ** 3) +
                 a4 * (K ** 4) +
                 a5 * (K ** 5))

            mask = X < 0
            w = w * ~mask + (1.0 - w) * mask

            return w

        def BS(X, R, V, T, K):
            """Black Scholes Function."""
            d1_arr = (np.log(X / K) + (R + V * V / 2.) * T) / (V * math.sqrt(T))
            d2_arr = d1_arr - V * math.sqrt(T)
            w_arr = CND(d1_arr)
            w2_arr = CND(d2_arr)
            return X * w_arr - X * math.exp(-R * T) * w2_arr

        # Recompute the CPU result only if necessary;
        start = System.nanoTime()
        if self.current_iter == 0 or reinit:
            res = BS(np.array(self.x_tmp), R, V, T, K)
            self.cpu_result = res[0]
        cpu_time = System.nanoTime() - start
        difference = np.abs(self.cpu_result - gpu_result)
        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", difference)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: {self.cpu_result:.4f}, " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


