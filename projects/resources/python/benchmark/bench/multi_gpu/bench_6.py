# coding=utf-8
import polyglot
import time
from java.lang import System
import numpy as np
from random import random, randint, seed

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D, DEFAULT_NUM_BLOCKS
from benchmark_result import BenchmarkResult

##############################
##############################

BLOCK_SIZE_V100 = 64  # Just a recommendation of optimal block size for the V100;
P = 16

NB_KERNEL = """   
extern "C" __global__ void nb_1(const int* x, const float* y, float* z, int n, int partition_rows, int n_feat, int n_classes, int partition_num) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < min(partition_rows, n - partition_num * partition_rows); i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_classes; j++) {
            for (int q = 0; q < n_feat; q++) {
                z[partition_num * partition_rows * n_classes + i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
            }
        }
    }
}
    
extern "C" __global__ void nb_2(const float* x, float* y, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float curr_max = x[i * n_col_x];
        for (int j = 0; j < n_col_x; j++) {
            curr_max = fmaxf(curr_max, x[i * n_col_x + j]);
        }
        y[i] = curr_max;
    }
}

extern "C" __global__ void nb_3(const float* x, const float* y, float* z, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < n_col_x; j++) {
            sum += expf(x[i * n_col_x + j] - y[i]);
        }
        z[i] = logf(sum) + y[i];
    }
}

extern "C" __global__ void nb_4(float* x, float* y, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] = expf(x[i * n_col_x + j] - y[i]);
        }
    }
}
"""

RR_KERNEL = """
extern "C" __global__ void rr_1(const int* x, float* mean, float *std, int n_row_x, int n_col_x, int partition, int partition_size) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {
        float feature_mean = 0;
        float sum_sq = 0;
        // Compute mean and variance;
        for (int i = 0; i < partition_size; i++) {
            float x_tmp = x[j * partition_size + i];
            feature_mean += x_tmp;
            sum_sq += x_tmp * x_tmp;
        }
        // feature_mean /= n_row_x;
        // std[j] = sqrtf(sum_sq / n_row_x - feature_mean * feature_mean);
        // mean[j] = feature_mean;

        // Keep just the sum and squared sum, compute mean and std later;
        mean[j] += feature_mean;
        std[j] += sum_sq;
    }
}

extern "C" __global__ void rr_1_1(float* mean, float *std, const float *mean_curr, const float *std_curr, int n_row_x, int n_col_x, int partition_index, int partition_size) {
    // We use partition 0 to accumulate, so skip it;
    if (partition_index == 0) return;

    // Aggregate mean and std from different partitions;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_col_x; i += blockDim.x * gridDim.x) {
        mean[i] += mean_curr[i];
        std[i] += std_curr[i];
        // When processing the last partition, compute the final mean and std;
        if (partition_index == %d - 1) {
            mean[i] /= n_row_x;
            std[i] = sqrtf(std[i] / n_row_x - mean[i] * mean[i]);
        }
    }
}

extern "C" __global__ void rr_1_2(const int *x, float *y, const float* mean, const float *std, int n_row_x, int n_col_x, int partition_size) {
    // Normalize each row;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < partition_size; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            float mean_curr = mean[j];
            float std_curr = std[j];
            y[i * n_col_x + j] = (x[i * n_col_x + j] - mean_curr) / std_curr;
        }
    }
}

extern "C" __global__ void rr_2(const float* x, const float* y, float* z, int n, int partition_rows, int n_feat, int n_classes, int partition_num) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < min(partition_rows, n - partition_num * partition_rows); i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_classes; j++) {
            for (int q = 0; q < n_feat; q++) {
                z[partition_num * partition_rows * n_classes + i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
            }
        }
    }
}

extern "C" __global__ void rr_3(float* x, const float* y, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] += y[j];
        }
    }
}
""" % (P)

ENSEMBLE_KERNEL = """
extern "C" __global__ void softmax(float* x, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float row_exp_sum = 0;
        for (int j = 0; j < n_col_x; j++) {
            row_exp_sum += expf(x[i * n_col_x + j]);
        }
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] = expf(x[i * n_col_x + j]) / row_exp_sum;
        }
    }
}

extern "C" __global__ void argmax(const float* x, const float* y, int* z, int n_row_x, int n_col_x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        int curr_best_index = 0;
        float curr_best = x[i * n_col_x] + y[i * n_col_x];
        for (int j = 0; j < n_col_x; j++) {
            float curr = x[i * n_col_x + j] + y[i * n_col_x + j];
            if (curr > curr_best) {
                curr_best = curr;
                curr_best_index = j;
            }
        }
        z[i] = curr_best_index;
    }
}
"""

##############################
##############################


class Benchmark6M(Benchmark):
    """
    Compute a an ensemble of Categorical Naive Bayes and Ridge Regression classifiers.
    Predictions are aggregated averaging the class scores after softmax normalization.
    The computation is done on mock data and parameters, but is conceptually identical to a real ML pipeline.
    In the DAG below, input arguments that are not involved in the computation of dependencies are omitted;

    RR-1: standard column normalization (partitioned row-wise)
        RR-1-1: aggregate mean/std across partitions (partitioned row-wise, but partitions are not independent)
        RR-1-2: apply normalization (partitioned row-wise)
    RR-2: matrix multiplication (partitioned row-wise)
    RR-3: add vector to matrix, row-wise
    NB-1: matrix multiplication (partitioned row-wise)
    NB-2: row-wise maximum
    NB-3: log of sum of exponential, row-wise
    NB-4: exponential, element-wise

         ┌─> RR-1(const X,MEAN,STD) ─> RR-1-1(MEAN,STD) -> RR-1-2(X, Z, MEAN, STD) ─> (...)
         │     (...) -> RR-2(const Z,R2) ─> RR-3(R2) ─> SOFTMAX(R1) ─────────────────────┐
        ─┤                                                                               ├─> ARGMAX(const R1,const R2,R)
         └─> NB-1(const X,R1) ─> NB-2(const R1,AMAX) ─> (...)                            │
               (...) -> NB-3(const R1,const AMAX,L) ─> NB-4(R1,const L) ─> SOFTMAX(R2) ──┘
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b6m", benchmark, nvprof_profile)
        self.size = 0
        self.S = 0
        self.x = [None for _ in range(P)]
        self.z = [None for _ in range(P)]
        self.mean = [None for _ in range(P)]
        self.std = [None for _ in range(P)]
        self.r1 = None
        self.r2 = None
        self.r = None

        self.nb_1 = None
        self.nb_2 = None
        self.nb_3 = None
        self.nb_4 = None
        self.rr_1 = None
        self.rr_1_1 = None
        self.rr_1_2 = None
        self.rr_2 = None
        self.rr_3 = None
        self.softmax = None
        self.argmax = None

        self.cpu_result = None
      
        # Internal arrays used by the algorithms, they do not affect the DAG structure;
        self.nb_feat_log_prob = None
        self.nb_class_log_prior = None
        self.ridge_coeff = None
        self.ridge_intercept = None
        self.nb_amax = None
        self.nb_l = None

        self.num_features = 1024 
        self.num_classes = 16  
        self.max_occurrence_of_ngram = 10

        self.num_blocks_size = self.num_blocks 
        self.num_blocks_feat = self.num_blocks 
        self.block_size = DEFAULT_BLOCK_SIZE_1D

        self.x_cpu = None
        self.nb_feat_log_prob_cpu = None
        self.ridge_coeff_cpu = None
        self.nb_class_log_prior_cpu = None
        self.ridge_intercept_cpu = None
        self.r1_cpu = None
        self.r2_cpu = None

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.S = (self.size + P - 1) // P
        self.block_size = block_size["block_size_1d"]

        # Allocate vectors;
        for i in range(P):
            self.x[i] = polyglot.eval(language="grcuda", string=f"int[{self.S * self.num_features}]")
            self.z[i] = polyglot.eval(language="grcuda", string=f"float[{self.S * self.num_features}]")
            self.mean[i] = polyglot.eval(language="grcuda", string=f"float[{self.num_features}]")
            self.std[i] = polyglot.eval(language="grcuda", string=f"float[{self.num_features}]")

        self.nb_feat_log_prob = polyglot.eval(language="grcuda", string=f"float[{self.num_classes * self.num_features}]")
        self.nb_class_log_prior = polyglot.eval(language="grcuda", string=f"float[{self.num_classes}]")
        self.ridge_coeff = polyglot.eval(language="grcuda", string=f"float[{self.num_classes * self.num_features}]")
        self.ridge_intercept = polyglot.eval(language="grcuda", string=f"float[{self.num_classes}]")

        self.nb_amax = polyglot.eval(language="grcuda", string=f"float[{self.size}]")
        self.nb_l = polyglot.eval(language="grcuda", string=f"float[{self.size}]")

        self.r1 = polyglot.eval(language="grcuda", string=f"float[{self.size * self.num_classes}]")
        self.r2 = polyglot.eval(language="grcuda", string=f"float[{self.size * self.num_classes}]")
        self.r = polyglot.eval(language="grcuda", string=f"int[{self.size}]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.nb_1 = build_kernel(NB_KERNEL, "nb_1", "const pointer, const pointer, const pointer, sint32, sint32, sint32, sint32, sint32")
        self.nb_2 = build_kernel(NB_KERNEL, "nb_2", "pointer, pointer, sint32, sint32")
        self.nb_3 = build_kernel(NB_KERNEL, "nb_3", "const pointer, const pointer, pointer, sint32, sint32")
        self.nb_4 = build_kernel(NB_KERNEL, "nb_4", "pointer, const pointer, sint32, sint32")

        self.rr_1 = build_kernel(RR_KERNEL, "rr_1", "const pointer, pointer, pointer, sint32, sint32, sint32, sint32")
        self.rr_1_1 = build_kernel(RR_KERNEL, "rr_1_1", "pointer, pointer, const pointer, const pointer, sint32, sint32, sint32, sint32")
        self.rr_1_2 = build_kernel(RR_KERNEL, "rr_1_2", "const pointer, pointer, const pointer, const pointer, sint32, sint32, sint32")
        self.rr_2 = build_kernel(RR_KERNEL, "rr_2", "const pointer, const pointer, const pointer, sint32, sint32, sint32, sint32, sint32")
        self.rr_3 = build_kernel(RR_KERNEL, "rr_3", "pointer, const pointer, sint32, sint32")

        self.softmax = build_kernel(ENSEMBLE_KERNEL, "softmax", "pointer, sint32, sint32")
        self.argmax = build_kernel(ENSEMBLE_KERNEL, "argmax", "const pointer, const pointer, pointer, sint32, sint32")
        self.initialize_rand = polyglot.eval(language="js", string="(x, m) => { for (let i = 0; i < x.length; i++) { x[i] = Math.floor(Math.random() * m) }}")

    @time_phase("initialization")
    def init(self):
        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)

        self.nb_feat_log_prob_cpu = np.random.random_sample((self.num_classes, self.num_features)).astype(dtype=np.float32)
        self.ridge_coeff_cpu = np.random.random_sample((self.num_classes, self.num_features)).astype(dtype=np.float32)
        self.nb_class_log_prior_cpu = np.random.random_sample(self.num_classes).astype(dtype=np.float32)
        self.ridge_intercept_cpu = np.random.random_sample(self.num_classes).astype(dtype=np.float32)

        for i in range(P):
            self.initialize_rand(self.x[i], self.max_occurrence_of_ngram)
        self.nb_feat_log_prob.copyFrom(int(np.int64(self.nb_feat_log_prob_cpu.ctypes.data)), len(self.nb_feat_log_prob))
        self.ridge_coeff.copyFrom(int(np.int64(self.ridge_coeff_cpu.ctypes.data)), len(self.ridge_coeff))
        self.nb_class_log_prior.copyFrom(int(np.int64(self.nb_class_log_prior_cpu.ctypes.data)), len(self.nb_class_log_prior))
        self.ridge_intercept.copyFrom(int(np.int64(self.ridge_intercept_cpu.ctypes.data)), len(self.ridge_intercept))

    @time_phase("reset_result")
    def reset_result(self) -> None:
        for i in range(self.size):
            for j in range(self.num_classes):
                self.r1[i * self.num_classes + j] = self.nb_class_log_prior[j]
                self.r2[i * self.num_classes + j] = 0
        for i in range(P):
            for j in range(self.num_features):
                self.mean[i][j] = 0.0
                self.std[i][j] = 0.0

    def execute(self) -> object:
        self.num_blocks_size = self.num_blocks
        self.num_blocks_feat = self.num_blocks
        self.block_size = self._block_size["block_size_1d"]
        # Schedule the categorical Naive Bayes and Ridge Regression kernels
        start_comp = System.nanoTime()
        start = 0

        # RR - 1.
        for i in range(P):
            self.execute_phase(f"rr_1_{i}", self.rr_1(self.num_blocks_feat, self.block_size),
                            self.x[i], self.mean[i], self.std[i], self.size, self.num_features, i, self.S)

        # RR - 1.1
        for i in range(P):
            self.execute_phase(f"rr_1_1_{i}", self.rr_1_1(self.num_blocks_feat, self.block_size),
                            self.mean[0], self.std[0], self.mean[i], self.std[i], self.size, self.num_features, i, self.S)

        # RR - 1.2 and 2.
        for i in range(P):
            self.execute_phase(f"rr_1_2_{i}", self.rr_1_2(self.num_blocks_feat, self.block_size),
                            self.x[i], self.z[i], self.mean[0], self.std[0], self.size, self.num_features, self.S)
            self.execute_phase(f"rr_2_{i}", self.rr_2(self.num_blocks_size, self.block_size),
                            self.z[i], self.ridge_coeff, self.r2, self.size, self.S, self.num_features, self.num_classes, i)

        # RR - 3.
        self.execute_phase("rr_3", self.rr_3(self.num_blocks_size, self.block_size),
                           self.r2, self.ridge_intercept, self.size, self.num_classes)

        # NB - 1.
        for i in range(P):
            self.execute_phase(f"nb_1_{i}", self.nb_1(self.num_blocks_size, self.block_size),
                            self.x[i], self.nb_feat_log_prob, self.r1, self.size, self.S, self.num_features, self.num_classes, i)

        # NB - 2.
        self.execute_phase("nb_2", self.nb_2(self.num_blocks_size, self.block_size),
                           self.r1, self.nb_amax, self.size, self.num_classes)

        # NB - 3.
        self.execute_phase("nb_3", self.nb_3(self.num_blocks_size, self.block_size),
                           self.r1, self.nb_amax, self.nb_l, self.size, self.num_classes)

        # NB - 4.
        self.execute_phase("nb_4", self.nb_4(self.num_blocks_size, self.block_size),
                           self.r1, self.nb_l, self.size, self.num_classes)

        # Ensemble results;

        # Softmax normalization;
        self.execute_phase("softmax_1", self.softmax(self.num_blocks_size, self.block_size), self.r1, self.size, self.num_classes)
        self.execute_phase("softmax_2", self.softmax(self.num_blocks_size, self.block_size), self.r2, self.size, self.num_classes)

        # Prediction;
        self.execute_phase("argmax", self.argmax(self.num_blocks_size, self.block_size), self.r1, self.r2, self.r, self.size, self.num_classes)

        # Add a final sync step to measure the real computation time;
        if self.time_phases:
            start = System.nanoTime()
        tmp = self.r[0]
        end = System.nanoTime()
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)
        self.benchmark.add_to_benchmark("gpu_result", 0)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: [" + ", ".join([f"{x:.4f}" for x in self.r[:10]]) + "...]")

        return self.r

    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:

        def softmax(X):
            return np.exp(X) / np.sum(np.exp(X), axis=1).reshape(X.shape[0], 1)

        def logsumexp(X):
            return np.log(np.sum(np.exp(X)))

        def naive_bayes_predict(X, feature_log_prob, log_class_prior):
            jll = X.dot(feature_log_prob.T) + log_class_prior
            amax = np.amax(jll, axis=1)
            l = logsumexp(jll - np.atleast_2d(amax).T) + amax

            return np.exp(jll - np.atleast_2d(l).T)

        def normalize(X):
            return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        def ridge_pred(X, coef, intercept):
            return np.dot(X, coef.T) + intercept

        # Recompute the CPU result only if necessary;
        start = System.nanoTime()
        if self.current_iter == 0 or reinit:
            # Re-initialize the random number generator with the same seed as the GPU to generate the same values;
            seed(self.random_seed)

            x_cpu = np.zeros((self.size, self.num_features), dtype=np.int32)
            for i in range(self.size):
                for j in range(self.num_features):
                    x_cpu[i, j] = self.x[i * self.num_features + j]

            r1_g = naive_bayes_predict(x_cpu, self.nb_feat_log_prob_cpu, self.nb_class_log_prior_cpu)
            r2_g = ridge_pred(normalize(x_cpu), self.ridge_coeff_cpu, self.ridge_intercept_cpu)
            r_g = np.argmax(softmax(r1_g) + softmax(r2_g), axis=1)
            self.cpu_result = r_g

        cpu_time = System.nanoTime() - start

        # Compare GPU and CPU results;
        difference = 0
        for i in range(self.size):
            difference += np.abs(self.cpu_result[i] - gpu_result[i])

        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", str(difference))
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: [" + ", ".join([f"{x:.4f}" for x in self.cpu_result[:10]]) + "...]; " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


