# coding=utf-8
import polyglot
import time
import numpy as np
from random import random, randint, seed, sample

from benchmark import Benchmark, time_phase
from benchmark_result import BenchmarkResult

##############################
##############################

NUM_THREADS_PER_BLOCK = 32

SPMV_KERNEL = """
    extern "C" __global__ void spmv(const int *ptr, const int *idx, const int *val, const float *vec, float *res, int num_rows, int num_nnz) {
    
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n < num_rows) {
            float sum = 0;
            for (int i = ptr[n]; i < ptr[n + 1]; i++) {
                sum += val[i] * vec[idx[i]];
            }
            res[n] = sum;
        }
    }
    """

SUM_KERNEL = """
    extern "C" __global__ void sum(const float *x, float *res, int n) {
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

DIVIDE_KERNEL = """
    extern "C" __global__ void divide(const float* x, float *y, const float *val, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            y[idx] = x[idx] / val[0];
        }
    }
    """

##############################
##############################


class Benchmark7(Benchmark):
    """
    Compute the HITS algorithm on a graph. The algorithm is composed of repeated sparse matrix-vector multiplications
    computed on a matrix and its transpose (outgoing and ingoing edges of a graph). The 2 matrix multiplications,
    for each iteration, can be computed in parallel;
    """

    def __init__(self, benchmark: BenchmarkResult):
        super().__init__("b7", benchmark)
        self.size = 0
        self.num_nnz = 0
        self.max_degree = 100  # Each vertex has 10 edges;
        self.num_iterations = 10
        self.ptr = None
        self.idx = None
        self.val = None
        self.ptr2 = None
        self.idx2 = None
        self.val2 = None
        self.auth1 = None
        self.auth2 = None
        self.hub1 = None
        self.hub2 = None
        self.auth_norm = None
        self.hub_norm = None

        self.ptr_cpu = None
        self.idx_cpu = None
        self.val_cpu = None
        self.ptr2_cpu = None
        self.idx2_cpu = None
        self.val2_cpu = None

        self.cpu_result = None
        self.gpu_result = None

        self.num_blocks_size = 0

        self.spmv_kernel = None
        self.sum_kernel = None
        self.divide_kernel = None

    @time_phase("allocation")
    def alloc(self, size: int):
        self.size = size
        self.num_nnz = size * self.max_degree
        self.num_blocks_size = (size + NUM_THREADS_PER_BLOCK - 1) // NUM_THREADS_PER_BLOCK

        self.gpu_result = np.zeros(self.size)

        # Allocate vectors;
        self.ptr = polyglot.eval(language="grcuda", string=f"int[{size + 1}]")
        self.ptr2 = polyglot.eval(language="grcuda", string=f"int[{size + 1}]")
        self.idx = polyglot.eval(language="grcuda", string=f"int[{self.num_nnz}]")
        self.idx2 = polyglot.eval(language="grcuda", string=f"int[{self.num_nnz}]")
        self.val = polyglot.eval(language="grcuda", string=f"int[{self.num_nnz}]")
        self.val2 = polyglot.eval(language="grcuda", string=f"int[{self.num_nnz}]")

        self.auth1 = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.auth2 = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.hub1 = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.hub2 = polyglot.eval(language="grcuda", string=f"float[{size}]")

        self.auth_norm = polyglot.eval(language="grcuda", string=f"float[1]")
        self.hub_norm = polyglot.eval(language="grcuda", string=f"float[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.spmv_kernel = build_kernel(SPMV_KERNEL, "spmv", "const pointer, const pointer, const pointer, const pointer, pointer, sint32, sint32")
        self.sum_kernel = build_kernel(SUM_KERNEL, "sum", "const pointer, pointer, sint32")
        self.divide_kernel = build_kernel(DIVIDE_KERNEL, "divide", "const pointer, pointer, const pointer, sint32")

    @time_phase("initialization")
    def init(self):

        def create_csr_from_coo(x_in, y_in, val_in):
            ptr_out = np.zeros(max(np.max(x_in), np.max(y_in)) + 2, dtype=int)
            for x_curr in x_in:
                ptr_out[x_curr + 1] += 1
            ptr_out = np.cumsum(ptr_out)
            return ptr_out, y_in, val_in

        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)

        # Create a random COO graph;
        x = []
        y = []
        val = []
        for i in range(self.size):
            # Create 10 random edges;
            edges = sorted(sample(range(self.size), self.max_degree))
            for j in edges:
                x += [i]
                y += [j]
                val += [1]

        # Turn the COO into CSR and CSC representations;
        self.ptr_cpu, self.idx_cpu, self.val_cpu = create_csr_from_coo(x, y, val)
        x2, y2 = zip(*sorted(zip(y, x)))
        self.ptr2_cpu, self.idx2_cpu, self.val2_cpu = create_csr_from_coo(x2, y2, val)
        for i in range(len(self.ptr_cpu)):
            self.ptr[i] = int(self.ptr_cpu[i])
            self.ptr2[i] = int(self.ptr2_cpu[i])
        for i in range(len(self.idx_cpu)):
            self.idx[i] = int(self.idx_cpu[i])
            self.idx2[i] = int(self.idx2_cpu[i])
            self.val[i] = int(self.val_cpu[i])
            self.val2[i] = int(self.val2_cpu[i])

    @time_phase("reset_result")
    def reset_result(self) -> None:
        for i in range(self.size):
            self.auth1[i] = 1.0
            self.auth2[i] = 1.0
            self.hub1[i] = 1.0
            self.hub2[i] = 1.0
        self.auth_norm[0] = 0.0
        self.hub_norm[0] = 0.0

    def execute(self) -> object:

        for i in range(self.num_iterations):
            # Authorities;
            start = time.time()
            self.spmv_kernel(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.ptr2, self.idx2, self.val2, self.hub1, self.auth2, self.size, self.num_nnz)
            end = time.time()
            self.benchmark.add_phase({"name": f"spmv_a_{i}", "time_sec": end - start})

            # Hubs;
            start = time.time()
            self.spmv_kernel(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.ptr, self.idx, self.val, self.auth1, self.hub2, self.size, self.num_nnz)
            end = time.time()
            self.benchmark.add_phase({"name": f"spmv_h_{i}", "time_sec": end - start})

            # Normalize authorities;
            start = time.time()
            self.sum_kernel(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.auth2, self.auth_norm, self.size)
            end = time.time()
            self.benchmark.add_phase({"name": f"sum_a_{i}", "time_sec": end - start})

            # Normalize hubs;
            start = time.time()
            self.sum_kernel(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.hub2, self.hub_norm, self.size)
            end = time.time()
            self.benchmark.add_phase({"name": f"sum_h_{i}", "time_sec": end - start})

            start = time.time()
            self.divide_kernel(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.auth2, self.auth1, self.auth_norm, self.size)
            end = time.time()
            self.benchmark.add_phase({"name": f"divide_a_{i}", "time_sec": end - start})

            start = time.time()
            self.divide_kernel(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.hub2, self.hub1, self.hub_norm, self.size)
            end = time.time()
            self.benchmark.add_phase({"name": f"divide_h_{i}", "time_sec": end - start})

        # Add a final sync step to measure the real computation time;
        start = time.time()
        tmp1 = self.auth1[0]
        tmp2 = self.hub1[0]
        end = time.time()
        self.benchmark.add_phase({"name": "sync", "time_sec": end - start})

        # Compute GPU result;
        for i in range(self.size):
            self.gpu_result[i] = self.auth1[i] + self.hub1[i]

        self.benchmark.add_to_benchmark("gpu_result", 0)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: [" + ", ".join([f"{x:.4f}" for x in self.gpu_result[:10]]) + "...]")

        return self.gpu_result

    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:

        def spmv(ptr, idx, val, vec):
            res = np.zeros(len(ptr) - 1)
            for i in range(len(ptr) - 1):
                curr_sum = 0
                start = int(ptr[i])
                end = int(ptr[i + 1])
                for j in range(start, end):
                    curr_sum += val[j] * vec[idx[j]]
                res[i] = curr_sum
            return res

        # Recompute the CPU result only if necessary;
        start = time.time()
        if self.current_iter == 0 or reinit:
            # Re-initialize the random number generator with the same seed as the GPU to generate the same values;
            seed(self.random_seed)
            # Initialize the support device arrays;
            N = self.size

            auth1 = np.ones(N)
            auth2 = np.ones(N)
            hub1 = np.ones(N)
            hub2 = np.ones(N)

            # Main iteration;
            for i in range(self.num_iterations):
                # Authority;
                auth2 = spmv(self.ptr2_cpu, self.idx2_cpu, self.val2_cpu, hub1)
                auth2 = auth2 / np.sum(auth2)
                # Hubs
                hub2 = spmv(self.ptr_cpu, self.idx_cpu, self.val_cpu, auth1)
                hub2 = hub2 / np.sum(hub2)

                auth1 = auth2
                hub1 = hub2
            self.cpu_result = hub1 + auth1

        cpu_time = time.time() - start

        # Compare GPU and CPU results;
        difference = 0
        for i in range(self.size):
            difference += np.abs(self.cpu_result[i] - gpu_result[i])

        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", str(difference))
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: [" + ", ".join([f"{x:.4f}" for x in self.cpu_result[:10]])
                                        + "...]; " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


