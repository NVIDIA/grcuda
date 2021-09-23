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
from random import random, randint, seed, sample

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D
from benchmark_result import BenchmarkResult

##############################
##############################

NUM_THREADS_PER_BLOCK = 32
THREADS_PER_VECTOR = 4
MAX_NUM_VECTORS_PER_BLOCK = 1024 / THREADS_PER_VECTOR

SPMV_KERNEL = """
extern "C" __global__ void spmv(const int *ptr, const int *idx, const int *val, const float *vec, float *res, int num_rows, int num_nnz) {

    for(int n = blockIdx.x * blockDim.x + threadIdx.x; n < num_rows; n += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int i = ptr[n]; i < ptr[n + 1]; i++) {
            sum += val[i] * vec[idx[i]];
        }
        res[n] = sum;
    }
}

extern "C" __global__ void spmv2(int* cudaRowCounter, int* d_ptr, int* d_cols, float* d_val, float* d_vector, float* d_out, int N) {
    int i;
    int thread_per_vector = %d;
    float sum;
    int row;
    int rowStart, rowEnd;
    int laneId = threadIdx.x %% thread_per_vector; //lane index in the vector
    int vectorId = threadIdx.x / thread_per_vector; //vector index in the thread block
    int warpLaneId = threadIdx.x & 31;	//lane index in the warp
    int warpVectorId = warpLaneId / thread_per_vector;	//vector index in the warp

    __shared__ volatile int space[%d][2];

    // Get the row index
    if (warpLaneId == 0) {
        row = atomicAdd(cudaRowCounter, 32 / thread_per_vector);
    }
    // Broadcast the value to other threads in the same warp and compute the row index of each vector
    row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

    while (row < N) {

        // Use two threads to fetch the row offset
        if (laneId < 2) {
            space[vectorId][laneId] = d_ptr[row + laneId];
        }
        rowStart = space[vectorId][0];
        rowEnd = space[vectorId][1];

        sum = 0;
        // Compute dot product
        if (thread_per_vector == 32) {

            // Ensure aligned memory access
            i = rowStart - (rowStart & (thread_per_vector - 1)) + laneId;

            // Process the unaligned part
            if (i >= rowStart && i < rowEnd) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }

                // Process the aligned part
            for (i += thread_per_vector; i < rowEnd; i += thread_per_vector) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        } else {
            for (i = rowStart + laneId; i < rowEnd; i += thread_per_vector) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        }
        // Intra-vector reduction
        for (i = thread_per_vector >> 1; i > 0; i >>= 1) {
            sum += __shfl_down_sync(0xffffffff,sum, i);
        }

        // Save the results
        if (laneId == 0) {
            d_out[row] = sum;
        }

        // Get a new row index
        if(warpLaneId == 0) {
            row = atomicAdd(cudaRowCounter, 32 / thread_per_vector);
        }
        // Broadcast the row index to the other threads in the same warp and compute the row index of each vector
        row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;
    }
}

// Compute d_out = y + alpha * A * d_vector;
extern "C" __global__ void spmv_full(int* cudaRowCounter, int* d_ptr, int* d_cols, float* d_val, float* d_vector, float* d_out, int N, float alpha, float* y) {
    int i;
    int thread_per_vector = %d;
    float sum;
    int row;
    int rowStart, rowEnd;
    int laneId = threadIdx.x %% thread_per_vector; //lane index in the vector
    int vectorId = threadIdx.x / thread_per_vector; //vector index in the thread block
    int warpLaneId = threadIdx.x & 31;	//lane index in the warp
    int warpVectorId = warpLaneId / thread_per_vector;	//vector index in the warp

    __shared__ volatile int space[%d][2];

    // Get the row index
    if (warpLaneId == 0) {
        row = atomicAdd(cudaRowCounter, 32 / thread_per_vector);
    }
    // Broadcast the value to other threads in the same warp and compute the row index of each vector
    row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

    while (row < N) {

        // Use two threads to fetch the row offset
        if (laneId < 2) {
            space[vectorId][laneId] = d_ptr[row + laneId];
        }
        rowStart = space[vectorId][0];
        rowEnd = space[vectorId][1];

        sum = 0;
        // Compute dot product
        if (thread_per_vector == 32) {

            // Ensure aligned memory access
            i = rowStart - (rowStart & (thread_per_vector - 1)) + laneId;

            // Process the unaligned part
            if (i >= rowStart && i < rowEnd) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }

                // Process the aligned part
            for (i += thread_per_vector; i < rowEnd; i += thread_per_vector) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        } else {
            for (i = rowStart + laneId; i < rowEnd; i += thread_per_vector) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        }
        // Intra-vector reduction
        for (i = thread_per_vector >> 1; i > 0; i >>= 1) {
            sum += __shfl_down_sync(0xffffffff,sum, i);
        }

        // Save the results
        if (laneId == 0) {
            d_out[row] = y[row] + alpha * sum;
        }

        // Get a new row index
        if(warpLaneId == 0) {
            row = atomicAdd(cudaRowCounter, 32 / thread_per_vector);
        }
        // Broadcast the row index to the other threads in the same warp and compute the row index of each vector
        row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;
    }
}
""" % (THREADS_PER_VECTOR, MAX_NUM_VECTORS_PER_BLOCK, THREADS_PER_VECTOR, MAX_NUM_VECTORS_PER_BLOCK)

SUM_KERNEL = """
__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void vector_norm(const float *x, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] * x[i];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

extern "C" __global__ void dot_product(const float *x, const float *y, float* z, int N) {
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
// Compute y = val + alpha * x;
extern "C" __global__ void saxpy(float* y, float *val, float *x, float alpha, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = val[i] + alpha * x[i];
    }
}

extern "C" __global__ void cpy(float *y, const float *x, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i];
    }
}
"""

##############################
##############################


class Benchmark9(Benchmark):
    """
    Compute the conjugate gradient algorithm on a sparse symmetric matrix.
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b9", benchmark, nvprof_profile)
        self.size = 0
        self.num_nnz = 0
        self.max_degree = 3  # Each row has 3 nnz entries (not counting symmetric entries);
        self.num_iterations = 4
        self.ptr = None
        self.idx = None
        self.val = None

        self.x = None
        self.b = None
        self.p = None
        self.r = None
        self.t1 = None
        self.t2 = None

        self.ptr_cpu = None
        self.idx_cpu = None
        self.val_cpu = None
        self.b_cpu = None

        self.cpu_result = None
        self.gpu_result = None

        self.num_blocks_size = 32
        self.block_size = None

        self.vspmv_kernel = None
        self.spmv_kernel = None
        self.norm_kernel = None
        self.saxpy_kernel = None

        self.row_cnt_1 = None
        self.row_cnt_2 = None
        self.row_cnt_3 = None

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.block_size = block_size["block_size_1d"]

        self.gpu_result = np.zeros(self.size)

        # Create a random symmetric COO matrix;
        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)

        # Create a random COO symmetric matrix;
        t = [(0,0,0)] * self.size * self.max_degree * 2
        for i in range(self.size):
            # Create max_degree random edges;
            edges = sample(range(0, self.size), self.max_degree)
            for j, e in enumerate(edges):
                while i == e:
                    e = randint(0, self.size - 1)
                tmp = random()
                t[i * self.max_degree + j] = (i, e, tmp)
                t[i * self.max_degree + j + self.size * self.max_degree] = (e, i, tmp)

        x, self.idx_cpu, self.val_cpu = zip(*sorted(t, key=lambda l: (l[0], l[1])))
        self.num_nnz = len(self.idx_cpu)
        self.ptr_cpu = [0] * (self.size + 1)
        for i, x_i in enumerate(x):
            self.ptr_cpu[x_i + 1] += 1
        for i in range(len(self.ptr_cpu) - 1):
            self.ptr_cpu[i + 1] += self.ptr_cpu[i]

        self.b_cpu = [0] * self.size

        # Allocate vectors;
        self.ptr = polyglot.eval(language="grcuda", string=f"int[{self.size + 1}]")
        self.idx = polyglot.eval(language="grcuda", string=f"int[{self.num_nnz}]")
        self.val = polyglot.eval(language="grcuda", string=f"float[{self.num_nnz}]")

        self.x = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.p = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.r = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.b = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.y = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.t1 = polyglot.eval(language="grcuda", string=f"float[1]")
        self.t2 = polyglot.eval(language="grcuda", string=f"float[1]")

        self.row_cnt_1 = polyglot.eval(language="grcuda", string=f"int[1]")
        self.row_cnt_2 = polyglot.eval(language="grcuda", string=f"int[1]")
        self.row_cnt_3 = polyglot.eval(language="grcuda", string=f"int[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.spmv_kernel = build_kernel(SPMV_KERNEL, "spmv2", "pointer, pointer, pointer, pointer, pointer, pointer, sint32")
        self.spmv_full_kernel = build_kernel(SPMV_KERNEL, "spmv_full", "pointer, pointer, pointer, pointer, pointer, pointer, sint32, float, pointer")
        self.norm_kernel = build_kernel(SUM_KERNEL, "vector_norm", "const pointer, pointer, sint32")
        self.dp_kernel = build_kernel(SUM_KERNEL, "dot_product", "const pointer, const pointer, pointer, sint32")
        self.saxpy_kernel = build_kernel(SAXPY_KERNEL, "saxpy", "pointer, const pointer, const pointer, float, sint32")
        self.cpy_kernel = build_kernel(SAXPY_KERNEL, "cpy", "pointer, const pointer, sint32")

    @time_phase("initialization")
    def init(self):
        for i in range(len(self.ptr_cpu)):
            self.ptr[i] = self.ptr_cpu[i]
        for i in range(len(self.idx_cpu)):
            self.idx[i] = self.idx_cpu[i]
            self.val[i] = self.val_cpu[i]
        for i in range(len(self.b)):
            self.b_cpu[i] = random()
            self.b[i] = self.b_cpu[i]

    @time_phase("reset_result")
    def reset_result(self) -> None:
        seed(self.random_seed)
        # Random initial solution;
        for i in range(self.size):
            self.x[i] = 1.0
        self.t1[0] = 0.0
        self.t2[0] = 0.0
        self.row_cnt_1[0] = 0
        self.row_cnt_2[0] = 0

    def execute(self) -> object:
        num_blocks_spmv = int(np.ceil(self.size / self.block_size))
        start_comp = System.nanoTime()
        start = 0

        # Initialization phase;
        # r = b - A * x
        self.execute_phase("spmv_init", self.spmv_full_kernel(num_blocks_spmv, self.block_size, 4 * self.block_size),
                           self.row_cnt_1, self.ptr, self.idx, self.val, self.x, self.r, self.size, -1, self.b)
        # p = r
        self.execute_phase("cpy_init", self.cpy_kernel(self.num_blocks_size, self.block_size), self.p, self.r, self.size)
        # t1 = r^t * r
        self.execute_phase("norm_init", self.norm_kernel(self.num_blocks_size, self.block_size), self.r, self.t1, self.size)

        for i in range(self.num_iterations):
            # t2 = p^t * A * p
            self.execute_phase(f"spmv_{i}", self.spmv_kernel(num_blocks_spmv, self.block_size, 4 * self.block_size),
                               self.row_cnt_2, self.ptr, self.idx, self.val, self.p, self.y, self.size)
            self.t2[0] = 0
            self.execute_phase(f"dp_{i}", self.dp_kernel(self.num_blocks_size, self.block_size), self.p, self.y, self.t2, self.size)

            if self.time_phases:
                start = System.nanoTime()
            alpha = self.t1[0] / self.t2[0]
            old_r_norm_squared = self.t1[0]
            self.t1[0] = 0
            self.row_cnt_1[0] = 0.0
            self.row_cnt_2[0] = 0.0
            if self.time_phases:
                end = System.nanoTime()
                self.benchmark.add_phase({"name": f"alpha_{i}", "time_sec": (end - start) / 1_000_000_000})

            # Update x: x = x + alpha * p
            self.execute_phase(f"saxpy_x_{i}", self.saxpy_kernel(self.num_blocks_size, self.block_size),
                               self.x, self.x, self.p, alpha, self.size)
            # r = r - alpha * y
            self.execute_phase(f"saxpy_r_{i}", self.saxpy_kernel(self.num_blocks_size, self.block_size),
                               self.r, self.r, self.y, -1 * alpha, self.size)
            # t1 = r^t * r
            self.execute_phase(f"norm_{i}", self.norm_kernel(self.num_blocks_size, self.block_size), self.r, self.t1, self.size)

            if self.time_phases:
                start = System.nanoTime()
            beta = self.t1[0] / old_r_norm_squared
            if self.time_phases:
                end = System.nanoTime()
                self.benchmark.add_phase({"name": f"beta_{i}", "time_sec": (end - start) / 1_000_000_000})

            self.execute_phase(f"saxpy_p_{i}", self.saxpy_kernel(self.num_blocks_size, self.block_size),
                               self.p, self.r, self.p, beta, self.size)

        # Add a final sync step to measure the real computation time;
        if self.time_phases:
            start = System.nanoTime()
        tmp1 = self.x[0]
        end = System.nanoTime()
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)
        # Compute GPU result;
        for i in range(self.size):
            self.gpu_result[i] = self.x[i]

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
        start = System.nanoTime()
        if self.current_iter == 0 or reinit:
            # Re-initialize the random number generator with the same seed as the GPU to generate the same values;
            seed(self.random_seed)
            # Initialize the support device arrays;
            N = self.size

            x = np.ones(N)
            # r = b - A * x
            r = np.array(self.b_cpu) - np.array(spmv(self.ptr_cpu, self.idx_cpu, self.val_cpu, x))
            p = r.copy()
            t1 = r.T.dot(r)

            # Main iteration;
            for i in range(self.num_iterations):
                y = spmv(self.ptr_cpu, self.idx_cpu, self.val_cpu, p)
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
        difference = 0
        for i in range(self.size):
            difference += np.abs(self.cpu_result[i] - gpu_result[i])

        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", str(difference))
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: [" + ", ".join([f"{x:.4f}" for x in self.cpu_result[:10]])
                                        + "...]; " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


