# coding=utf-8
import polyglot
from java.lang import System
import numpy as np
from random import random, randint, seed, sample
import os
import pickle
import json

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D, DEFAULT_NUM_BLOCKS
from benchmark_result import BenchmarkResult

##############################
##############################

STORE_TO_PICKLE = True
LOAD_FROM_PICKLE = False
LOAD_FROM_JSON = True

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

extern "C" __global__ void spmv3(int* cudaRowCounter, int* d_ptr, int* d_cols, int* d_val, float* d_vector, float* d_out, int N) {
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
""" % (THREADS_PER_VECTOR, MAX_NUM_VECTORS_PER_BLOCK)


SUM_KERNEL = """
__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void sum(float *x, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}
"""

DIVIDE_KERNEL = """
extern "C" __global__ void divide(float* x, float *y, float *val, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] / val[0];
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

     ┌─> SPMV(const H1,A2) ┬─> SUM(const A2,A_norm) ┬─> DIVIDE(A1,const A2,const A_norm) ─> CPU: A_norm=0 ─> (repeat)
     │                     └─────────┐              │
    ─┤                     ┌─────────│──────────────┘
     │                     │         └──────────────┐
     └─> SPMV(const A1,H2) ┴─> SUM(const H2,H_norm) ┴─> DIVIDE(H1,const H2,const H_norm) ─> CPU: H_norm=0 ─> (repeat)
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b7", benchmark, nvprof_profile)
        self.size = 0
        self.num_nnz = 0
        self.max_degree = 3  # Each vertex has 3 edges;
        self.num_iterations = 5
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
        self.row_cnt_1 = None
        self.row_cnt_2 = None

        self.ptr_cpu = None
        self.idx_cpu = None
        self.val_cpu = None
        self.ptr2_cpu = None
        self.idx2_cpu = None
        self.val2_cpu = None

        self.cpu_result = None
        self.gpu_result = None

        self.num_blocks_size = DEFAULT_NUM_BLOCKS
        self.block_size = None

        self.spmv_kernel = None
        self.sum_kernel = None
        self.divide_kernel = None

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.num_nnz = size * self.max_degree
        self.block_size = block_size["block_size_1d"]

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

        self.row_cnt_1 = polyglot.eval(language="grcuda", string=f"int[1]")
        self.row_cnt_2 = polyglot.eval(language="grcuda", string=f"int[1]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.spmv_kernel = build_kernel(SPMV_KERNEL, "spmv3",
                                        "pointer, pointer, pointer, pointer, pointer, pointer, sint32")
        self.sum_kernel = build_kernel(SUM_KERNEL, "sum", "pointer, pointer, sint32")
        self.divide_kernel = build_kernel(DIVIDE_KERNEL, "divide", "pointer, pointer, pointer, sint32")

    @time_phase("initialization")
    def init(self):

        def get_pickle_filename(size: int) -> str:
            pickle_folder = f"{os.getenv('GRCUDA_HOME')}/data/pickle"
            if not os.path.exists(pickle_folder):
                os.makedirs(pickle_folder)
            return os.path.join(pickle_folder, f"graph_{size}")

        def get_json_filename(size: int) -> str:
            pickle_folder = f"{os.getenv('GRCUDA_HOME')}/data/json"
            if not os.path.exists(pickle_folder):
                os.makedirs(pickle_folder)
            return os.path.join(pickle_folder, f"graph_{size}.json")

        def create_csr_from_coo(x_in, y_in, val_in, size, degree=None):
            if degree:
                ptr_out = [degree] * (size + 1)
                ptr_out[0] = 0
            else:
                # values, count = np.unique(x_in, return_counts=True)
                ptr_out = [0] * (size + 1)
                for x in x_in:
                    ptr_out[x] += 1
            for i in range(len(ptr_out) - 1):
                ptr_out[i + 1] += ptr_out[i]
            # ptr_out = np.cumsum(ptr_out, dtype=np.int32)
            return ptr_out, y_in, val_in

        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)

        if LOAD_FROM_PICKLE:
            start = System.nanoTime()
            pickle_file_name = get_pickle_filename(self.size)
            if self.benchmark.debug:
                BenchmarkResult.log_message(f"load pickled data from {pickle_file_name}")
            with open(pickle_file_name, "rb") as f:
                [self.ptr_cpu, self.idx_cpu, self.val_cpu, self.ptr2_cpu, self.idx2_cpu, self.val2_cpu] = pickle.load(f)
            print(f"load time pickle = {(System.nanoTime() - start) / 1_000_000_000}")
        elif LOAD_FROM_JSON:
            start = System.nanoTime()
            json_file_name = get_json_filename(self.size)
            if self.benchmark.debug:
                BenchmarkResult.log_message(f"load json data from {json_file_name}")
            with open(json_file_name, "rb") as f:
                import codecs
                graph = json.load(codecs.getreader('utf-8')(f))
                self.ptr_cpu = graph["ptr_cpu"]
                self.idx_cpu = graph["idx_cpu"]
                self.val_cpu = graph["val_cpu"]
                self.ptr2_cpu = graph["ptr2_cpu"]
                self.idx2_cpu = graph["idx2_cpu"]
                self.val2_cpu = graph["val2_cpu"]
            print(f"load time json = {(System.nanoTime() - start) / 1_000_000_000}")
        else:
            # Create a random COO graph;
            x = [0] * self.size * self.max_degree
            y = [0] * self.size * self.max_degree
            val = [1] * self.size * self.max_degree
            for i in range(self.size):
                # Create max_degree random edges;
                edges = sorted(sample(range(self.size), self.max_degree))
                for j, e in enumerate(edges):
                    x[i * self.max_degree + j] = i
                    y[i * self.max_degree + j] = e

            # Turn the COO into CSR and CSC representations;
            self.ptr_cpu, self.idx_cpu, self.val_cpu = create_csr_from_coo(x, y, val, self.size, degree=self.max_degree)
            x2, y2 = zip(*sorted(zip(y, x)))
            self.ptr2_cpu, self.idx2_cpu, self.val2_cpu = create_csr_from_coo(x2, y2, val, self.size)

            # Low-level copies from numpy array, they are faster but require slow casting to numpy arrays;
            # self.ptr.copyFrom(int(np.int64(self.ptr_cpu.ctypes.data)), len(self.ptr))
            # self.ptr2.copyFrom(int(np.int64(self.ptr2_cpu.ctypes.data)), len(self.ptr2))
            # self.idx.copyFrom(int(np.int64(self.idx_cpu.ctypes.data)), len(self.idx))
            # self.idx2.copyFrom(int(np.int64(self.idx2_cpu.ctypes.data)), len(self.idx2))
            # self.val.copyFrom(int(np.int64(self.val_cpu.ctypes.data)), len(self.val))
            # self.val2.copyFrom(int(np.int64(self.val2_cpu.ctypes.data)), len(self.val2))

            # for i in range(len(self.ptr_cpu)):
            #     self.ptr[i] = self.ptr_cpu[i]
            #     self.ptr2[i] = self.ptr2_cpu[i]
            # for i in range(len(self.idx_cpu)):
            #     self.idx[i] = self.idx_cpu[i]
            #     self.idx2[i] = self.idx2_cpu[i]
            #     self.val[i] = self.val_cpu[i]
            #     self.val2[i] = self.val2_cpu[i]
            if STORE_TO_PICKLE:
                pickle_file_name = get_pickle_filename(self.size)
                if self.benchmark.debug:
                    BenchmarkResult.log_message(f"store pickled data to {pickle_file_name}")
                with open(pickle_file_name, "wb+") as f:
                    pickle.dump([self.ptr_cpu, self.idx_cpu, self.val_cpu, self.ptr2_cpu, self.idx2_cpu, self.val2_cpu], f)

    @time_phase("reset_result")
    def reset_result(self) -> None:
        for i in range(len(self.ptr_cpu)):
            self.ptr[i] = self.ptr_cpu[i]
            self.ptr2[i] = self.ptr2_cpu[i]
        for i in range(len(self.idx_cpu)):
            self.idx[i] = self.idx_cpu[i]
            self.idx2[i] = self.idx2_cpu[i]
            self.val[i] = self.val_cpu[i]
            self.val2[i] = self.val2_cpu[i]
        for i in range(self.size):
            self.auth1[i] = 1.0
            self.auth2[i] = 1.0
            self.hub1[i] = 1.0
            self.hub2[i] = 1.0
        self.auth_norm[0] = 0.0
        self.hub_norm[0] = 0.0
        self.row_cnt_1[0] = 0
        self.row_cnt_2[0] = 0

    def execute(self) -> object:

        num_blocks_spmv = int(np.ceil(self.size / self.block_size))

        start_comp = System.nanoTime()
        start = 0

        for i in range(self.num_iterations):
            # Authorities;
            self.execute_phase(f"spmv_a_{i}", self.spmv_kernel(num_blocks_spmv, self.block_size, 4 * self.block_size), self.row_cnt_1, self.ptr2,
                               self.idx2, self.val2, self.hub1, self.auth2, self.size)

            # Hubs;
            self.execute_phase(f"spmv_h_{i}", self.spmv_kernel(num_blocks_spmv, self.block_size, 4 * self.block_size), self.row_cnt_2, self.ptr,
                               self.idx, self.val, self.auth1, self.hub2, self.size)

            # Normalize authorities;
            self.execute_phase(f"sum_a_{i}", self.sum_kernel(self.num_blocks_size, self.block_size), self.auth2,
                               self.auth_norm, self.size)

            # Normalize hubs;
            self.execute_phase(f"sum_h_{i}", self.sum_kernel(self.num_blocks_size, self.block_size), self.hub2,
                               self.hub_norm, self.size)

            self.execute_phase(f"divide_a_{i}", self.divide_kernel(self.num_blocks_size, self.block_size), self.auth2,
                               self.auth1, self.auth_norm, self.size)

            self.execute_phase(f"divide_h_{i}", self.divide_kernel(self.num_blocks_size, self.block_size), self.hub2,
                               self.hub1, self.hub_norm, self.size)

            if self.time_phases:
                start = System.nanoTime()
            self.auth_norm[0] = 0.0
            self.hub_norm[0] = 0.0
            self.row_cnt_1[0] = 0.0
            self.row_cnt_2[0] = 0.0
            if self.time_phases:
                end = System.nanoTime()
                self.benchmark.add_phase({"name": f"norm_reset_{i}", "time_sec": (end - start) / 1_000_000_000})

        # Add a final sync step to measure the real computation time;
        if self.time_phases:
            start = System.nanoTime()
        tmp1 = self.auth1[0]
        tmp2 = self.hub1[0]
        end = System.nanoTime()
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)
        # Compute GPU result;
        for i in range(self.size):
            self.gpu_result[i] = self.auth1[i] + self.hub1[i]

        self.benchmark.add_to_benchmark("gpu_result", 0)
        if self.benchmark.debug:
            BenchmarkResult.log_message(
                f"\tgpu result: [" + ", ".join([f"{x:.4f}" for x in self.gpu_result[:10]]) + "...]")

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

            auth1 = np.ones(N)
            hub1 = np.ones(N)

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
