# coding=utf-8
import polyglot
import time
import numpy as np
from random import random, randint, seed

from benchmark import Benchmark, time_phase
from benchmark_result import BenchmarkResult

##############################
##############################

NUM_THREADS_PER_BLOCK = 1024

NB_KERNEL = """   
    extern "C" __global__ void nb_1(const int* x, const float* y, float* z, int size, int n_feat, int n_classes) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            for (int j = 0; j < n_classes; j++) {
                for (int q = 0; q < n_feat; q++) {
                    z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
                }
            }
        }
    }
    
    extern "C" __global__ void nb_2(const float* x, float* y, int n_row_x, int n_col_x) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_row_x) {
            float curr_max = x[i * n_col_x];
            for (int j = 0; j < n_col_x; j++) {
                curr_max = fmaxf(curr_max, x[i * n_col_x + j]); 
            }
            y[i] = curr_max;
        }
    }
    
    extern "C" __global__ void nb_3(const float* x, const float* y, float* z, int n_row_x, int n_col_x) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0;
        if (i < n_row_x) {
            for (int j = 0; j < n_col_x; j++) {
                sum += expf(x[i * n_col_x + j] - y[i]);
            }
            z[i] = logf(sum) + y[i];
        }
    }
    
    extern "C" __global__ void nb_4(float* x, float* y, int n_row_x, int n_col_x) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_row_x) {
            for (int j = 0; j < n_col_x; j++) {
                x[i * n_col_x + j] = expf(x[i * n_col_x + j] - y[i]);
            }
        }
    }
    """

RR_KERNEL = """
    extern "C" __global__ void rr_1(const int* x, float *y, int n_row_x, int n_col_x) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j < n_col_x) {
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
    
    extern "C" __global__ void rr_2(const float* x, const float* y, float* z, int size, int n_feat, int n_classes) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            for (int j = 0; j < n_classes; j++) {
                for (int q = 0; q < n_feat; q++) {
                    z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
                }
            }
        }
    }

    extern "C" __global__ void rr_3(float* x, const float *y, int n_row_x, int n_col_x) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_row_x) {
            for (int j = 0; j < n_col_x; j++) {
                x[i * n_col_x + j] += y[j];
            }
        }
    }
    """

ENSEMBLE_KERNEL = """
    extern "C" __global__ void softmax(float *x, int n_row_x, int n_col_x) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_row_x) {
            float row_exp_sum = 0;
            for (int j = 0; j < n_col_x; j++) {
                row_exp_sum += expf( x[i * n_col_x + j]);
            }
            for (int j = 0; j < n_col_x; j++) {
                 x[i * n_col_x + j] = expf(x[i * n_col_x + j]) / row_exp_sum;
            }
        }
    }
    
    extern "C" __global__ void argmax(const float *x, const float *y, int *z, int n_row_x, int n_col_x) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_row_x) {
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
    Compute a complex pipeline of kernels, doing mock computations, and using read-only arguments;
                         ┌─> B(const Y, R1) ───────────────────┐
        A: (const X, Y) ─┤                                     ├─> E(const R1, const R2, R)
                         └─> C(const Y, Z) ─> D(const Z, R2) ──┘
    """

    def __init__(self, benchmark: BenchmarkResult):
        super().__init__("b6", benchmark)
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

        self.num_features = 1000  # self.nb_feat_log_prob_np.shape[1]
        self.num_classes = 5  # self.nb_feat_log_prob_np.shape[0]

        self.num_blocks_size = 0
        self.num_blocks_feat = (self.num_features + NUM_THREADS_PER_BLOCK - 1) // NUM_THREADS_PER_BLOCK

    @time_phase("allocation")
    def alloc(self, size: int):
        self.size = size
        self.num_blocks_size = (size + NUM_THREADS_PER_BLOCK - 1) // NUM_THREADS_PER_BLOCK

        # Allocate vectors;
        self.x = polyglot.eval(language="grcuda", string=f"int[{size}][{self.num_features}]")
        self.z = polyglot.eval(language="grcuda", string=f"float[{size}][{self.num_features}]")

        self.nb_feat_log_prob = polyglot.eval(language="grcuda", string=f"float[{self.num_classes}][{self.num_features}]")
        self.nb_class_log_prior = polyglot.eval(language="grcuda", string=f"float[{self.num_classes}]")
        self.ridge_coeff = polyglot.eval(language="grcuda", string=f"float[{self.num_classes}][{self.num_features}]")
        self.ridge_intercept = polyglot.eval(language="grcuda", string=f"float[{self.num_classes}]")

        self.nb_amax = polyglot.eval(language="grcuda", string=f"float[{self.size}]")
        self.nb_l = polyglot.eval(language="grcuda", string=f"float[{self.size}]")

        self.r1 = polyglot.eval(language="grcuda", string=f"float[{self.size}][{self.num_classes}]")
        self.r2 = polyglot.eval(language="grcuda", string=f"float[{self.size}][{self.num_classes}]")
        self.r = polyglot.eval(language="grcuda", string=f"int[{self.size}]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.nb_1 = build_kernel(NB_KERNEL, "nb_1", "const pointer, const pointer, pointer, sint32, sint32, sint32")
        self.nb_2 = build_kernel(NB_KERNEL, "nb_2", "const pointer, pointer, sint32, sint32")
        self.nb_3 = build_kernel(NB_KERNEL, "nb_3", "const pointer, const pointer, pointer, sint32, sint32")
        self.nb_4 = build_kernel(NB_KERNEL, "nb_4", "pointer, const pointer, sint32, sint32")

        self.rr_1 = build_kernel(RR_KERNEL, "rr_1", "const pointer, pointer, sint32, sint32")
        self.rr_2 = build_kernel(RR_KERNEL, "rr_2", "const pointer, const pointer, pointer, sint32, sint32, sint32")
        self.rr_3 = build_kernel(RR_KERNEL, "rr_3", "pointer, const pointer, sint32, sint32")

        self.softmax = build_kernel(ENSEMBLE_KERNEL, "softmax", "pointer, sint32, sint32")
        self.argmax = build_kernel(ENSEMBLE_KERNEL, "argmax", "const pointer, const pointer, pointer, sint32, sint32")

    @time_phase("initialization")
    def init(self):
        self.random_seed = randint(0, 10000000)
        seed(self.random_seed)

        # Initialize the support device arrays;
        for i in range(self.num_classes):
            for j in range(self.num_features):
                self.nb_feat_log_prob[i][j] = random()  # self.nb_feat_log_prob_np[i][j]
                self.ridge_coeff[i][j] = random()  # self.ridge_coeff_np[i][j]
            self.nb_class_log_prior[i] = random()  # self.nb_class_log_prior_np[i]
            self.ridge_intercept[i] = random()  # self.ridge_intercept_np[i]

        # Create a random input;
        max_occurrence_of_ngram = 10
        for i in range(self.size):
            for j in range(self.num_features):
                self.x[i][j] = randint(0, max_occurrence_of_ngram)
            # Initialize a support array;
            for j in range(self.num_classes):
                self.r1[i][j] = self.nb_class_log_prior[j]
                self.r2[i][j] = 0

    @time_phase("reset_result")
    def reset_result(self) -> None:
        for i in range(self.size):
            for j in range(self.num_classes):
                self.r1[i][j] = self.nb_class_log_prior[j]
                self.r2[i][j] = 0

    def execute(self) -> object:

        # Schedule the categorical Naive Bayes and Ridge Regression kernels

        # RR - 1.
        start = time.time()
        self.rr_1(self.num_blocks_feat, NUM_THREADS_PER_BLOCK)(self.x, self.z, self.size, self.num_features)
        end = time.time()
        self.benchmark.add_phase({"name": "rr_1", "time_sec": end - start})

        # NB - 1.
        start = time.time()
        self.nb_1(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.x, self.nb_feat_log_prob, self.r1, self.size, self.num_features, self.num_classes)
        end = time.time()
        self.benchmark.add_phase({"name": "nb_1", "time_sec": end - start})

        # RR - 2.
        start = time.time()
        self.rr_2(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.z, self.ridge_coeff, self.r2, self.size, self.num_features, self.num_classes)
        end = time.time()
        self.benchmark.add_phase({"name": "rr_2", "time_sec": end - start})

        # NB - 2.
        start = time.time()
        self.nb_2(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.r1, self.nb_amax, self.size, self.num_classes)
        end = time.time()
        self.benchmark.add_phase({"name": "nb_2", "time_sec": end - start})

        # NB - 3.
        start = time.time()
        self.nb_3(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.r1, self.nb_amax, self.nb_l, self.size, self.num_classes)
        end = time.time()
        self.benchmark.add_phase({"name": "nb_3", "time_sec": end - start})

        # RR - 3.
        start = time.time()
        self.rr_3(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.r2, self.ridge_intercept, self.size, self.num_classes)
        end = time.time()
        self.benchmark.add_phase({"name": "rr_3", "time_sec": end - start})

        # NB - 4.
        start = time.time()
        self.nb_4(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.r1, self.nb_l, self.size, self.num_classes)
        end = time.time()
        self.benchmark.add_phase({"name": "nb_4", "time_sec": end - start})

        # FIXME: this causes an NPE in java!
        # for i in range(5):
        #     print(self.r1[i][0])
        # for i in range(5):
        #     print(self.r2[i][0])

        # Ensemble results;

        # Softmax normalization;
        start = time.time()
        self.softmax(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.r1, self.size, self.num_classes)
        self.softmax(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.r2, self.size, self.num_classes)
        end = time.time()
        self.benchmark.add_phase({"name": "softmax", "time_sec": end - start})

        # Prediction;
        start = time.time()
        self.argmax(self.num_blocks_size, NUM_THREADS_PER_BLOCK)(self.r1, self.r2, self.r, self.size, self.num_classes)
        end = time.time()
        self.benchmark.add_phase({"name": "argmax", "time_sec": end - start})

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
        start = time.time()
        if self.current_iter == 0 or reinit:
            # Re-initialize the random number generator with the same seed as the GPU to generate the same values;
            seed(self.random_seed)
            # Initialize the support device arrays;
            x_g = np.zeros((self.size, self.num_features))
            feat_log_prob = np.zeros((self.num_classes, self.num_features))
            ridge_coeff = np.zeros((self.num_classes, self.num_features))
            class_log_prior = np.zeros(self.num_classes)
            ridge_intercept = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                for j in range(self.num_features):
                    feat_log_prob[i, j] = random()  # self.nb_feat_log_prob_np[i][j]
                    ridge_coeff[i, j] = random()  # self.ridge_coeff_np[i][j]
                class_log_prior[i] = random()  # self.nb_class_log_prior_np[i]
                ridge_intercept[i] = random()  # self.ridge_intercept_np[i]

            # Create a random input;
            for i in range(self.size):
                for j in range(self.num_features):
                    x_g[i, j] = self.x[i][j]

            r1_g = naive_bayes_predict(x_g, feat_log_prob, class_log_prior)
            r2_g = ridge_pred(normalize(x_g), ridge_coeff, ridge_intercept)
            r_g = np.argmax(softmax(r1_g) + softmax(r2_g), axis=1)
            self.cpu_result = r_g

        cpu_time = time.time() - start

        # Compare GPU and CPU results;
        difference = 0
        for i in range(self.size):
            difference += np.abs(self.cpu_result[i] - gpu_result[i])

        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", str(difference))
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: [" + ", ".join([f"{x:.4f}" for x in self.cpu_result[:10]]) + "...]; " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")


