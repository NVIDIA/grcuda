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
import time
from java.lang import System
import numpy as np
from random import random, randint, seed

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D, DEFAULT_NUM_BLOCKS
from benchmark_result import BenchmarkResult

##############################
##############################

NB_KERNEL = """   
    extern "C" __global__ void nb_1(const int* x, float* y, float* z, int size, int n_feat, int n_classes) {
        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
            for (int j = 0; j < n_classes; j++) {
                for (int q = 0; q < n_feat; q++) {
                    z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
                } 
            }
        }
    }
    
    extern "C" __global__ void nb_2(float* x, float* y, int n_row_x, int n_col_x) {
        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
            float curr_max = x[i * n_col_x];
            for (int j = 0; j < n_col_x; j++) {
                curr_max = fmaxf(curr_max, x[i * n_col_x + j]); 
            }
            y[i] = curr_max;
        }
    }
    
    extern "C" __global__ void nb_3(float* x, float* y, float* z, int n_row_x, int n_col_x) {
        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
            float sum = 0;
            for (int j = 0; j < n_col_x; j++) {
                sum += expf(x[i * n_col_x + j] - y[i]);
            }
            z[i] = logf(sum) + y[i];
        }
    }
    
    extern "C" __global__ void nb_4(float* x, float* y, int n_row_x, int n_col_x) {
        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
            for (int j = 0; j < n_col_x; j++) {
                x[i * n_col_x + j] = expf(x[i * n_col_x + j] - y[i]);
            }
        }
    }
    """

RR_KERNEL = """
    extern "C" __global__ void rr_1(const int* x, float *y, int n_row_x, int n_col_x) {
        for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {
            float feature_mean = 0;
            float sum_sq = 0;
            // Compute mean and variance;
            for (int i = 0; i < n_row_x; i++) {
                feature_mean += x[j * n_row_x + i];
                sum_sq += x[j * n_row_x + i] * x[j * n_row_x + i];
            }
            feature_mean /= n_row_x;
            float std = sqrtf(sum_sq / n_row_x - feature_mean * feature_mean);
            
            // Update values;
            for (int i = 0; i < n_row_x; i++) {
                y[j * n_row_x + i] = ((float) x[j * n_row_x + i] - feature_mean) / std;
            }
        }
    }
    
    extern "C" __global__ void rr_2(float* x, float* y, float* z, int size, int n_feat, int n_classes) {
        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
            for (int j = 0; j < n_classes; j++) {
                for (int q = 0; q < n_feat; q++) {
                    z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
                }
            }
        }
    }

    extern "C" __global__ void rr_3(float* x, float *y, int n_row_x, int n_col_x) {
        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
            for (int j = 0; j < n_col_x; j++) {
                x[i * n_col_x + j] += y[j];
            }
        }
    }
    """

ENSEMBLE_KERNEL = """
    extern "C" __global__ void softmax(float *x, int n_row_x, int n_col_x) {
        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
            float row_exp_sum = 0;
            for (int j = 0; j < n_col_x; j++) {
                row_exp_sum += expf( x[i * n_col_x + j]);
            }
            for (int j = 0; j < n_col_x; j++) {
                 x[i * n_col_x + j] = expf(x[i * n_col_x + j]) / row_exp_sum;
            }
        }
    }
    
    extern "C" __global__ void argmax(float *x, float *y, int *z, int n_row_x, int n_col_x) {
        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
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


class Benchmark6(Benchmark):
    """
    Compute an ensemble of Categorical Naive Bayes and Ridge Regression classifiers. 
    Predictions are aggregated averaging the class scores after softmax normalization.
    The computation is done on mock data and parameters, but is conceptually identical to a real ML pipeline.
    In the DAG below, input arguments that are not involved in the computation of dependencies are omitted.

    The size of the benchmark is the number of rows in the matrix (each representing a document with 200 features).
    Predictions are done by choosing among 10 classes.
    The Ridge Regression classifier takes about 2x the time of the Categorical Naive Bayes classifier.

    Structure of the computation:

    RR-1: standard normalization
    RR-2: matrix multiplication
    RR-3: add vector to matrix, row-wise
    NB-1: matrix multiplication
    NB-2: row-wise maximum
    NB-3: log of sum of exponential, row-wise
    NB-4: exponential, element-wise

     ┌─> RR-1(const X,Z) ─> RR-2(const Z,R2) ─> RR-3(R2) ─> SOFTMAX(R1) ─────────────┐
    ─┤                                                                               ├─> ARGMAX(const R1,const R2,R)
     └─> NB-1(const X,R1) ─> NB-2(const R1,AMAX) ─> (...)                            │
           (...) -> NB-3(const R1,const AMAX,L) ─> NB-4(R1,const L) ─> SOFTMAX(R2) ──┘
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b6", benchmark, nvprof_profile)
        self.size = 0
        self.x = None
        self.z = None
        self.r1 = None
        self.r2 = None
        self.r = None

        self.nb_1 = None
        self.nb_2 = None
        self.nb_3 = None
        self.nb_4 = None
        self.rr_1 = None
        self.rr_2 = None
        self.rr_3 = None
        self.softmax = None
        self.argmax = None

        self.cpu_result = None

        # Load matrices from files;
        # self.nb_feat_log_prob_np = np.loadtxt("../other/data/nb_feat_log_prob.csv", delimiter=",")
        # self.nb_class_log_prior_np = np.loadtxt("../other/data/nb_class_log_prior.csv", delimiter=",")
        # self.ridge_coeff_np = np.loadtxt("../other/data/ridge_coeff.csv", delimiter=",")
        # self.ridge_intercept_np = np.loadtxt("../other/data/ridge_intercept.csv", delimiter=",")

        # Internal arrays used by the algorithms, they do not affect the DAG structure;
        self.nb_feat_log_prob = None
        self.nb_class_log_prior = None
        self.ridge_coeff = None
        self.ridge_intercept = None
        self.nb_amax = None
        self.nb_l = None

        self.num_features = 200  # self.nb_feat_log_prob_np.shape[1]
        self.num_classes = 10  # self.nb_feat_log_prob_np.shape[0]

        self.num_blocks_size = self.num_blocks # 64  # DEFAULT_NUM_BLOCKS
        self.num_blocks_feat = self.num_blocks # 64  # DEFAULT_NUM_BLOCKS
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
        self.block_size = block_size["block_size_1d"]

        # Allocate vectors;
        self.x = polyglot.eval(language="grcuda", string=f"int[{size * self.num_features}]")
        self.z = polyglot.eval(language="grcuda", string=f"float[{size * self.num_features}]")

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
        self.nb_1 = build_kernel(NB_KERNEL, "nb_1", "const pointer, pointer, pointer, sint32, sint32, sint32")
        self.nb_2 = build_kernel(NB_KERNEL, "nb_2", "pointer, pointer, sint32, sint32")
        self.nb_3 = build_kernel(NB_KERNEL, "nb_3", "pointer, pointer, pointer, sint32, sint32")
        self.nb_4 = build_kernel(NB_KERNEL, "nb_4", "pointer, pointer, sint32, sint32")

        self.rr_1 = build_kernel(RR_KERNEL, "rr_1", "const pointer, pointer, sint32, sint32")
        self.rr_2 = build_kernel(RR_KERNEL, "rr_2", "pointer, pointer, pointer, sint32, sint32, sint32")
        self.rr_3 = build_kernel(RR_KERNEL, "rr_3", "pointer, pointer, sint32, sint32")

        self.softmax = build_kernel(ENSEMBLE_KERNEL, "softmax", "pointer, sint32, sint32")
        self.argmax = build_kernel(ENSEMBLE_KERNEL, "argmax", "pointer, pointer, pointer, sint32, sint32")

    @time_phase("initialization")
    def init(self):
        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)

        # Create a random input;
        max_occurrence_of_ngram = 10
        self.x_cpu = np.random.randint(0, max_occurrence_of_ngram, (self.size, self.num_features), dtype=np.int32)

        self.nb_feat_log_prob_cpu = np.random.random_sample((self.num_classes, self.num_features)).astype(dtype=np.float32)
        self.ridge_coeff_cpu = np.random.random_sample((self.num_classes, self.num_features)).astype(dtype=np.float32)
        self.nb_class_log_prior_cpu = np.random.random_sample(self.num_classes).astype(dtype=np.float32)
        self.ridge_intercept_cpu = np.random.random_sample(self.num_classes).astype(dtype=np.float32)

        self.r1_cpu = np.zeros((self.size, self.num_classes))
        for j in range(self.num_classes):
            self.r1_cpu[:, j] = self.nb_class_log_prior_cpu[j]
        self.r2_cpu = np.zeros((self.size, self.num_classes))

        self.x.copyFrom(int(np.int64(self.x_cpu.ctypes.data)), len(self.x))
        self.nb_feat_log_prob.copyFrom(int(np.int64(self.nb_feat_log_prob_cpu.ctypes.data)), len(self.nb_feat_log_prob))
        self.ridge_coeff.copyFrom(int(np.int64(self.ridge_coeff_cpu.ctypes.data)), len(self.ridge_coeff))
        self.nb_class_log_prior.copyFrom(int(np.int64(self.nb_class_log_prior_cpu.ctypes.data)), len(self.nb_class_log_prior))
        self.ridge_intercept.copyFrom(int(np.int64(self.ridge_intercept_cpu.ctypes.data)), len(self.ridge_intercept))
        self.r1.copyFrom(int(np.int64(self.r1_cpu.ctypes.data)), len(self.r1))
        self.r2.copyFrom(int(np.int64(self.r2_cpu.ctypes.data)), len(self.r2))

    @time_phase("reset_result")
    def reset_result(self) -> None:
        for i in range(self.size):
            for j in range(self.num_classes):
                self.r1[i * self.num_classes + j] = self.nb_class_log_prior[j]
                self.r2[i * self.num_classes + j] = 0

    def execute(self) -> object:
        self.num_blocks_size = self.num_blocks  # 64  # DEFAULT_NUM_BLOCKS
        self.num_blocks_feat = self.num_blocks  # 64  # DEFAULT_NUM_BLOCKS
        self.block_size = self._block_size["block_size_1d"]
        # Schedule the categorical Naive Bayes and Ridge Regression kernels
        start_comp = System.nanoTime()
        start = 0

        # RR - 1.
        self.execute_phase("rr_1", self.rr_1(self.num_blocks_feat, self.block_size),
                           self.x, self.z, self.size, self.num_features)

        # NB - 1.
        self.execute_phase("nb_1", self.nb_1(self.num_blocks_size, self.block_size),
                           self.x, self.nb_feat_log_prob, self.r1, self.size, self.num_features, self.num_classes)

        # RR - 2.
        self.execute_phase("rr_2", self.rr_2(self.num_blocks_size, self.block_size),
                           self.z, self.ridge_coeff, self.r2, self.size, self.num_features, self.num_classes)

        # NB - 2.
        self.execute_phase("nb_2", self.nb_2(self.num_blocks_size, self.block_size),
                           self.r1, self.nb_amax, self.size, self.num_classes)

        # NB - 3.
        self.execute_phase("nb_3", self.nb_3(self.num_blocks_size, self.block_size),
                           self.r1, self.nb_amax, self.nb_l, self.size, self.num_classes)

        # RR - 3.
        self.execute_phase("rr_3", self.rr_3(self.num_blocks_size, self.block_size),
                           self.r2, self.ridge_intercept, self.size, self.num_classes)

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

            r1_g = naive_bayes_predict(self.x_cpu, self.nb_feat_log_prob_cpu, self.nb_class_log_prior_cpu)
            r2_g = ridge_pred(normalize(self.x_cpu), self.ridge_coeff_cpu, self.ridge_intercept_cpu)
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


