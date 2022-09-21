/*
 * Copyright (c) 2022 NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package it.necst.grcuda.benchmark.bench;

import it.necst.grcuda.benchmark.Benchmark;
import it.necst.grcuda.benchmark.BenchmarkConfig;
import org.graalvm.polyglot.Value;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

// Just a recommendation of optimal block size for the V100 (BLOCK_SIZE_V100 = 64)
public class B6M extends Benchmark {
    static final int P = 16;

    private static final String NB_KERNEL = "" +
            "extern \"C\" __global__ void nb_1(const int* x, const float* y, float* z, int n, int partition_rows, int n_feat, int n_classes, int partition_num) {\n" +
            "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < min(partition_rows, n - partition_num * partition_rows); i += blockDim.x * gridDim.x) {\n" +
            "        for (int j = 0; j < n_classes; j++) {\n" +
            "            for (int q = 0; q < n_feat; q++) {\n" +
            "                z[partition_num * partition_rows * n_classes + i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];\n" +
            "            }\n" +
            "        }\n" +
            "    }\n" +
            "}\n" +
            "    \n" +
            "extern \"C\" __global__ void nb_2(const float* x, float* y, int n_row_x, int n_col_x) {\n" +
            "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
            "        float curr_max = x[i * n_col_x];\n" +
            "        for (int j = 0; j < n_col_x; j++) {\n" +
            "            curr_max = fmaxf(curr_max, x[i * n_col_x + j]);\n" +
            "        }\n" +
            "        y[i] = curr_max;\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "extern \"C\" __global__ void nb_3(const float* x, const float* y, float* z, int n_row_x, int n_col_x) {\n" +
            "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
            "        float sum = 0;\n" +
            "        for (int j = 0; j < n_col_x; j++) {\n" +
            "            sum += expf(x[i * n_col_x + j] - y[i]);\n" +
            "        }\n" +
            "        z[i] = logf(sum) + y[i];\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "extern \"C\" __global__ void nb_4(float* x, float* y, int n_row_x, int n_col_x) {\n" +
            "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
            "        for (int j = 0; j < n_col_x; j++) {\n" +
            "            x[i * n_col_x + j] = expf(x[i * n_col_x + j] - y[i]);\n" +
            "        }\n" +
            "    }\n" +
            "}";

    private static final String RR_KERNEL = "" +
            "extern \"C\" __global__ void rr_1(const int* x, float* mean, float *std, int n_row_x, int n_col_x, int partition, int partition_size) {\n" +
            "    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {\n" +
            "        float feature_mean = 0;\n" +
            "        float sum_sq = 0;\n" +
            "        // Compute mean and variance;\n" +
            "        for (int i = 0; i < partition_size; i++) {\n" +
            "            float x_tmp = x[j * partition_size + i];\n" +
            "            feature_mean += x_tmp;\n" +
            "            sum_sq += x_tmp * x_tmp;\n" +
            "        }\n" +
            "        // feature_mean /= n_row_x;\n" +
            "        // std[j] = sqrtf(sum_sq / n_row_x - feature_mean * feature_mean);\n" +
            "        // mean[j] = feature_mean;\n" +
            "\n" +
            "        // Keep just the sum and squared sum, compute mean and std later;\n" +
            "        mean[j] += feature_mean;\n" +
            "        std[j] += sum_sq;\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "extern \"C\" __global__ void rr_1_1(float* mean, float *std, const float *mean_curr, const float *std_curr, int n_row_x, int n_col_x, int partition_index, int partition_size) {\n" +
            "    // We use partition 0 to accumulate, so skip it;\n" +
            "    if (partition_index == 0) return;\n" +
            "\n" +
            "    // Aggregate mean and std from different partitions;\n" +
            "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_col_x; i += blockDim.x * gridDim.x) {\n" +
            "        mean[i] += mean_curr[i];\n" +
            "        std[i] += std_curr[i];\n" +
            "        // When processing the last partition, compute the final mean and std;\n" +
            "        if (partition_index == " + P + "- 1) {\n" +
            "            mean[i] /= n_row_x;\n" +
            "            std[i] = sqrtf(std[i] / n_row_x - mean[i] * mean[i]);\n" +
            "        }\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "extern \"C\" __global__ void rr_1_2(const int *x, float *y, const float* mean, const float *std, int n_row_x, int n_col_x, int partition_size) {\n" +
            "    // Normalize each row;\n" +
            "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < partition_size; i += blockDim.x * gridDim.x) {\n" +
            "        for (int j = 0; j < n_col_x; j++) {\n" +
            "            float mean_curr = mean[j];\n" +
            "            float std_curr = std[j];\n" +
            "            y[i * n_col_x + j] = (x[i * n_col_x + j] - mean_curr) / std_curr;\n" +
            "        }\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "extern \"C\" __global__ void rr_2(const float* x, const float* y, float* z, int n, int partition_rows, int n_feat, int n_classes, int partition_num) {\n" +
            "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < min(partition_rows, n - partition_num * partition_rows); i += blockDim.x * gridDim.x) {\n" +
            "        for (int j = 0; j < n_classes; j++) {\n" +
            "            for (int q = 0; q < n_feat; q++) {\n" +
            "                z[partition_num * partition_rows * n_classes + i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];\n" +
            "            }\n" +
            "        }\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "extern \"C\" __global__ void rr_3(float* x, const float* y, int n_row_x, int n_col_x) {\n" +
            "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
            "        for (int j = 0; j < n_col_x; j++) {\n" +
            "            x[i * n_col_x + j] += y[j];\n" +
            "        }\n" +
            "    }\n" +
            "}";

    private static final String ENSEMBLE_KERNEL = "" +
            "extern \"C\" __global__ void softmax(float* x, int n_row_x, int n_col_x) {\n" +
            "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
            "        float row_exp_sum = 0;\n" +
            "        for (int j = 0; j < n_col_x; j++) {\n" +
            "            row_exp_sum += expf(x[i * n_col_x + j]);\n" +
            "        }\n" +
            "        for (int j = 0; j < n_col_x; j++) {\n" +
            "            x[i * n_col_x + j] = expf(x[i * n_col_x + j]) / row_exp_sum;\n" +
            "        }\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "extern \"C\" __global__ void argmax(const float* x, const float* y, int* z, int n_row_x, int n_col_x) {\n" +
            "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
            "        int curr_best_index = 0;\n" +
            "        float curr_best = x[i * n_col_x] + y[i * n_col_x];\n" +
            "        for (int j = 0; j < n_col_x; j++) {\n" +
            "            float curr = x[i * n_col_x + j] + y[i * n_col_x + j];\n" +
            "            if (curr > curr_best) {\n" +
            "                curr_best = curr;\n" +
            "                curr_best_index = j;\n" +
            "            }\n" +
            "        }\n" +
            "        z[i] = curr_best_index;\n" +
            "    }\n" +
            "}";

    /*
    Compute an ensemble of Categorical Naive Bayes and Ridge Regression classifiers.
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
     */

    private Value[] x, z, mean, std;
    private Value nb_1, nb_2, nb_3, nb_4;
    private Value rr_1, rr_1_1, rr_1_2, rr_2, rr_3;
    private Value argmax, softmax;
    private Value initialize_rand;
    private Value nb_feat_log_prob, nb_class_log_prior;
    private Value ridge_coeff, ridge_intercept;
    private Value nb_amax, nb_l;
    private Value r1, r2, r;
    private int S;
    private int num_features;
    private int num_classes;
    private int max_occurrence_of_ngram;

    private List<List<List<Integer>>> x_cpu;
    private float[][] nb_feat_log_prob_cpu;
    private float[][] ridge_coeff_cpu;
    private float nb_class_log_prior_cpu;
    private float ridge_intercept_cpu;

    public B6M(BenchmarkConfig currentConfig) {
        super(currentConfig);

        this.S = 0;

        x = new Value[P];
        z = new Value[P];
        mean = new Value[P];
        std = new Value[P];


        /*for (int i = 0; i < P; i++) {
            this.z[i] = null;
            this.mean[i] = null;
            this.std[i] = null;
        }*/
        this.r1 = null;
        this.r2 = null;
        this.r = null;

        this.nb_1 = null;
        this.nb_2 = null;
        this.nb_3 = null;
        this.nb_4 = null;
        this.rr_1 = null;
        this.rr_1_1 = null;
        this.rr_1_2 = null;
        this.rr_2 = null;
        this.rr_3 = null;
        this.softmax = null;
        this.argmax = null;

        // Internal arrays used by the algorithms, they do not affect the DAG structure
        this.nb_feat_log_prob = null;
        this.nb_class_log_prior = null;
        this.ridge_coeff = null;
        this.ridge_intercept = null;
        this.nb_amax = null;
        this.nb_l = null;

        this.num_features = 1024;
        this.num_classes = 16;
        this.max_occurrence_of_ngram = 10;

        this.x_cpu = null;
        this.nb_feat_log_prob_cpu =  null;
        this.ridge_coeff_cpu = null;
        this.nb_class_log_prior_cpu = 0;
        this.ridge_intercept_cpu = 0;
    }

    @Override
    public void allocateTest(int iteration) {
        this.S = Math.floorDiv(config.size + P - 1, P);

        // Allocate vectors
        for (int i = 0; i < P; i++) {
            this.x[i] = requestArray("int", this.S * this.num_features);
            this.z[i] = requestArray("float", this.S * this.num_features);
            this.mean[i] = requestArray("float", this.num_features);
            this.std[i] = requestArray("float", this.num_features);
        }

        this.nb_feat_log_prob = requestArray("float", this.num_classes * this.num_features);
        this.nb_class_log_prior = requestArray("float", this.num_classes);
        this.ridge_coeff = requestArray("float", this.num_classes * this.num_features);
        this.ridge_intercept = requestArray("float", this.num_classes);

        this.nb_amax = requestArray("float", config.size);
        this.nb_l = requestArray("float", config.size);

        this.r1 = requestArray("float", config.size * this.num_classes);
        this.r2 = requestArray("float", config.size * this.num_classes);
        this.r = requestArray("int", config.size);

        // Build the kernels
        Value buildKernel = context.eval("grcuda", "buildkernel");
        this.nb_1 = buildKernel.execute(NB_KERNEL, "nb_1", "const pointer, const pointer, const pointer, sint32, sint32, sint32, sint32, sint32");
        this.nb_2 = buildKernel.execute(NB_KERNEL, "nb_2", "pointer, pointer, sint32, sint32");
        this.nb_3 = buildKernel.execute(NB_KERNEL, "nb_3", "const pointer, const pointer, pointer, sint32, sint32");
        this.nb_4 = buildKernel.execute(NB_KERNEL, "nb_4", "pointer, const pointer, sint32, sint32");

        this.rr_1 = buildKernel.execute(RR_KERNEL, "rr_1", "const pointer, pointer, pointer, sint32, sint32, sint32, sint32");
        this.rr_1_1 = buildKernel.execute(RR_KERNEL, "rr_1_1", "pointer, pointer, const pointer, const pointer, sint32, sint32, sint32, sint32");
        this.rr_1_2 = buildKernel.execute(RR_KERNEL, "rr_1_2", "const pointer, pointer, const pointer, const pointer, sint32, sint32, sint32");
        this.rr_2 = buildKernel.execute(RR_KERNEL, "rr_2", "const pointer, const pointer, const pointer, sint32, sint32, sint32, sint32, sint32");
        this.rr_3 = buildKernel.execute(RR_KERNEL, "rr_3", "pointer, const pointer, sint32, sint32");

        this.softmax = buildKernel.execute(ENSEMBLE_KERNEL, "softmax", "pointer, sint32, sint32");
        this.argmax = buildKernel.execute(ENSEMBLE_KERNEL, "argmax", "const pointer, const pointer, pointer, sint32, sint32");
        this.initialize_rand = context.eval("js", "(x, m) => { for (let i = 0; i < x.length; i++) { x[i] = Math.floor(Math.random() * m) }}");
    }


    @Override
    public void initializeTest(int iteration) {
        assert (!config.randomInit); // randomInit not supported yet
        // Random init not optional
        Random random = new Random(System.currentTimeMillis());

        for (int i = 0; i < P; i++)
            this.initialize_rand.execute(this.x[i], this.max_occurrence_of_ngram);

        for(int i=0; i<nb_feat_log_prob.getArraySize(); i++)
            this.nb_feat_log_prob.setArrayElement(i, random.nextFloat());

        for(int i=0; i<ridge_coeff.getArraySize(); i++)
            this.ridge_coeff.setArrayElement(i, random.nextFloat());

        for(int i=0; i<nb_class_log_prior.getArraySize(); i++)
            this.nb_class_log_prior.setArrayElement(i, random.nextFloat());

        for(int i=0; i<ridge_intercept.getArraySize(); i++)
            this.ridge_intercept.setArrayElement(i, random.nextFloat());
    }


    @Override
    public void resetIteration(int iteration) {
        for (int i = 0; i < config.size; i++) {
            for (int j = 0; j < this.num_classes; j++) {
                this.r1.setArrayElement((long) i * this.num_classes + j, this.nb_class_log_prior.getArrayElement(j));
                this.r2.setArrayElement((long) i * this.num_classes + j, 0);
            }
        }
        for (int i = 0; i < P; i++) {
            for (int j = 0; j < this.num_features; j++) {
                this.mean[i].setArrayElement(j, 0.0);
                this.std[i].setArrayElement(j, 0.0);
            }
        }
    }

    @Override
    public void runTest(int iteration) {
        long start = System.nanoTime();

        // Schedule the categorical Naive Bayes and Ridge Regression kernels

        // RR - 1.
        for (int i = 0; i < P; i++) {
            this.rr_1.execute(config.numBlocks, config.blockSize1D) // Set parameters
                    .execute(this.x[i], this.mean[i], this.std[i], this.config.size, this.num_features, i, this.S);
        }

        // RR - 1.1
        for (int i = 0; i < P; i++) {
            this.rr_1_1.execute(config.numBlocks, config.blockSize1D) // Set parameters
                    .execute(this.mean[0], this.std[0], this.mean[i], this.std[i], this.config.size, this.num_features, i, this.S);
        }

        // RR - 1.2 and 2.
        for (int i = 0; i < P; i++) {
            this.rr_1_2.execute(config.numBlocks, config.blockSize1D) // Set parameters
                    .execute(this.x[i], this.z[i], this.mean[0], this.std[0], this.config.size, this.num_features, this.S);
            this.rr_2.execute(config.numBlocks, config.blockSize1D) // Set parameters
                    .execute(this.z[i], this.ridge_coeff, this.r2, this.config.size, this.S, this.num_features, this.num_classes, i);
        }

        // RR - 3.
        this.rr_3.execute(config.numBlocks, config.blockSize1D) // Set parameters
                .execute(this.r2, this.ridge_intercept, this.config.size, this.num_classes);

        // NB - 1.
        for (int i = 0; i < P; i++) {
            this.nb_1.execute(config.numBlocks, config.blockSize1D) // Set parameters
                    .execute(this.x[i], this.nb_feat_log_prob, this.r1, this.config.size, this.S, this.num_features, this.num_classes, i);
        }

        // NB - 2.
        this.nb_2.execute(config.numBlocks, config.blockSize1D) // Set parameters
                .execute(this.r1, this.nb_amax, this.config.size, this.num_classes);

        // NB - 3.
        this.nb_3.execute(config.numBlocks, config.blockSize1D) // Set parameters
                .execute(this.r1, this.nb_amax, this.nb_l, this.config.size, this.num_classes);

        // NB - 4.
        this.nb_4.execute(config.numBlocks, config.blockSize1D) // Set parameters
                .execute(this.r1, this.nb_l, this.config.size, this.num_classes);

        // Ensemble results

        // Softmax normalization;
        this.softmax.execute(config.numBlocks, config.blockSize1D) // Set parameters
                .execute(this.r1, this.config.size, this.num_classes);
        this.softmax.execute(config.numBlocks, config.blockSize1D) // Set parameters
                .execute(this.r2, this.config.size, this.num_classes);

        // Prediction
        this.argmax.execute(config.numBlocks, config.blockSize1D) // Set parameters
                .execute(this.r1, this.r2, this.r, this.config.size, this.num_classes);

        // Sync step to measure the real computation time
        int tmp = this.r.getArrayElement(0).asInt();
        long end = System.nanoTime();
        benchmarkResults.setCurrentGpuResult(0);
        benchmarkResults.setCurrentComputationSec((end-start)/1000000000F);
    }

    @Override
    public void cpuValidation() {
        // Re-initialize the random number generator with the same seed as the GPU to generate the same values;
        assert (!config.randomInit);
        x_cpu = new ArrayList<>(new ArrayList<>(new ArrayList<>()));

        for (int r = 0; r < config.size; r++) {
            for (int c = 0; c < this.num_classes; c++) {
                x_cpu.get(r).get(c).add(0);
            }
        }

        for (int r = 0; r < config.size; r++) {
            for (int c = 0; c < this.num_classes; c++) {
                for (int i = 0; i < this.S * this.num_features; i++) {
                    x_cpu.get(r).get(c).add(this.x[r * this.num_features + c].getArrayElement(i).asInt());
                }
            }
        }

        float[] r_g;

        // TODO:
        /*
            r1_g = naive_bayes_predict(x_cpu, self.nb_feat_log_prob_cpu, self.nb_class_log_prior_cpu)
            r2_g = ridge_pred(normalize(x_cpu), self.ridge_coeff_cpu, self.ridge_intercept_cpu)
            r_g = np.argmax(softmax(r1_g) + softmax(r2_g), axis=1)
            self.cpu_result = r_g

             # Compare GPU and CPU results;
            difference = 0
            for i in range(self.size):
            difference += np.abs(self.cpu_result[i] - gpu_result[i])

         */
    }

    private float[] softmax(float[] X, int n_col_x, int n_row_x) {
        float row_exp_sum = 0;
        float[] result = new float[X.length];

        for (int r=0; r<n_row_x; r++) {
            for (int c = 0; c < n_col_x; c++) {
                row_exp_sum += Math.exp(X[r + n_col_x + c]);
            }

            for (int c = 0; c < n_col_x; c++) {
                result[r * n_col_x + c] = (float) (Math.exp(X[r * n_col_x + c]) / row_exp_sum);
            }
        }

        return result;
    }

    private void logsumexp() {
        /*
            return np.log(np.sum(np.exp(X)))
         */
    }

    private void naive_bayes_predict() {
        /*
            jll = X.dot(feature_log_prob.T) + log_class_prior
            amax = np.amax(jll, axis=1)
            l = logsumexp(jll - np.atleast_2d(amax).T) + amax

            return np.exp(jll - np.atleast_2d(l).T)
         */
    }

    private void normalize(float[][] X) {
        /*
            return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
         */
    }

    private float[] ridge_pred(float[] X, float[] coef, float[] intercept) {
        /*
            return np.dot(X, coef.T) + intercept
         */
        float[] result = new float[intercept.length];
        float dotResult = dotProduct(X, coef);

        for (int i = 0; i<intercept.length; i++) {
            result[i] = dotResult + intercept[i];
        }
        return null;
    }

    public static float dotProduct(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
