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

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class B9M extends Benchmark {
    /*
    Compute the conjugate gradient algorithm on a dense symmetric matrix.
    The matrix-vector multiplications are row-partitioned to scale across multiple GPUs;
     */

    private static final String PRECONDITION_KERNEL = "" +
            "// Add a small epsilon to the main diagonal:\n" +
            "extern \"C\" __global__ void precondition(float *A, int n, int m, int offset) {\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {\n" +
            "        A[i * n + i + offset] += 1e-12; \n" +
            "    }\n" +
            "}";

    private static final String MMUL_KERNEL = "" +
            "// z = x @ y;\n" +
            "extern \"C\" __global__ void matrix_vector_mult(const float* x, const float* y, float* z, int n, int m, int z_offset) {\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        float sum = 0;\n" +
            "        for (int j = 0; j < m; j++) {                \n" +
            "            sum += x[i * m + j] * y[j];\n" +
            "        }\n" +
            "        z[z_offset + i] = sum;\n" +
            "    }\n" +
            "}\n" +
            "// z := w + alpha * A @ y;\n" +
            "extern \"C\" __global__ void matrix_vector_mult_axpy(const float* x, const float* y, const float *w, const float alpha, float* z, int n, int m, int z_offset) {\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        float sum = 0;\n" +
            "        for (int j = 0; j < m; j++) {                \n" +
            "            sum += x[i * m + j] * y[j];\n" +
            "        }\n" +
            "        z[z_offset + i] = alpha * sum + w[z_offset + i];\n" +
            "    }\n" +
            "}";

    private static final String DP_KERNEL = "" +
            "__inline__ __device__ float warp_reduce(float val) {\n" +
            "    int warp_size = 32;\n" +
            "    for (int offset = warp_size / 2; offset > 0; offset /= 2) \n" +
            "        val += __shfl_down_sync(0xFFFFFFFF, val, offset);\n" +
            "    return val;\n" +
            "}\n" +
            "// z = <x, x>;\n" +
            "extern \"C\" __global__ void l2_norm(const float *x, float* z, int N) {\n" +
            "    int warp_size = 32;\n" +
            "    float sum = float(0);\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {\n" +
            "        float x_tmp = x[i];\n" +
            "        sum += x_tmp * x_tmp;\n" +
            "    }\n" +
            "    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;\n" +
            "    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster\n" +
            "        atomicAdd(z, sum); // The first thread in the warp updates the output;\n" +
            "}\n" +
            "// z = <x, y>;\n" +
            "extern \"C\" __global__ void dot(const float *x, const float *y, float* z, int N) {\n" +
            "    int warp_size = 32;\n" +
            "    float sum = float(0);\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {\n" +
            "        sum += x[i] * y[i];\n" +
            "    }\n" +
            "    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;\n" +
            "    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster\n" +
            "        atomicAdd(z, sum); // The first thread in the warp updates the output;\n" +
            "}";

    private static final String SAXPY_KERNEL = "" +
            "// y = val + alpha * x;\n" +
            "extern \"C\" __global__ void saxpy(float* y, const float *val, const float *x, float alpha, int n) {\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        y[i] = val[i] + alpha * x[i];\n" +
            "    }\n" +
            "}\n" +
            "// Simply copy array x into y;\n" +
            "extern \"C\" __global__ void cpy(float *y, const float *x, int n) {\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        y[i] = x[i];\n" +
            "    }\n" +
            "}";

    private Value precondition_kernel, mmul_kernel, mmul_axpy_kernel, l2_norm_kernel, dp_kernel, saxpy_kernel, copy_kernel, initialize_random_symmetric_matrix;
    private Value[] A;
    private Value x, b, p, r, y, t1, t2;
    private int S;

    private final int P = 16;
    private final int ITER = 50;

    public B9M(BenchmarkConfig currentConfig) {
        super(currentConfig);

        this.S = 0;
        this.A = new Value[this.P];
        for (int i = 0; i < this.P; i ++) this.A[i] = null;
        this.x = null;
        this.b = null;
        this.p = null;
        this.r = null;
        this.y = null;
        this.t1 = null;
        this.t2 = null;

        this.mmul_axpy_kernel = null;
        this.mmul_kernel = null;
        this.l2_norm_kernel = null;
        this.dp_kernel = null;
        this.saxpy_kernel = null;
        this.copy_kernel = null;
    }

    @Override
    public void allocateTest(int iteration) {
        this.S = Math.floorDiv(config.size + this.P - 1, this.P);

        // Allocate vectors
        for (int i = 0; i < this.P; i++)
            this.A[i] = requestArray("float", this.S * config.size);
        this.x = requestArray("float", config.size);
        this.b = requestArray("float", config.size);
        this.p = requestArray("float", config.size);
        this.r = requestArray("float", config.size);
        this.y = requestArray("float", config.size);
        this.t1 = requestArray("float", 1);
        this.t2 = requestArray("float", 1);

        // Build the kernels
        Value buildKernel = context.eval("grcuda", "buildkernel");

        this.precondition_kernel = buildKernel.execute(PRECONDITION_KERNEL, "precondition", "pointer, sint32, sint32, sint32");
        this.mmul_kernel = buildKernel.execute(MMUL_KERNEL, "matrix_vector_mult", "const pointer, const pointer, const pointer, sint32, sint32, sint32");
        this.mmul_axpy_kernel = buildKernel.execute(MMUL_KERNEL, "matrix_vector_mult_axpy", "const pointer, const pointer, pointer, float, const pointer, sint32, sint32, sint32");
        this.l2_norm_kernel = buildKernel.execute(DP_KERNEL, "l2_norm", "const pointer, pointer, sint32");
        this.dp_kernel = buildKernel.execute(DP_KERNEL, "dot", "const pointer, pointer, pointer, sint32");
        this.saxpy_kernel = buildKernel.execute(SAXPY_KERNEL, "saxpy", "pointer, const pointer, const pointer, float, sint32");
        this.copy_kernel = buildKernel.execute(SAXPY_KERNEL, "cpy", "pointer, pointer, sint32");
        this.initialize_random_symmetric_matrix = context.eval("js", "(X, S, N) => { \n" +
                "            for (let i = 0; i < N; i++) {\n" +
                "                s = (i / S) >> 0;\n" +
                "                k = i % S;\n" +
                "                Xs = X[s];\n" +
                "                i_N = k * N;\n" +
                "                for (let j = i; j < N; j++) {\n" +
                "                    val = 2 * Math.random() - 1; \n" +
                "                    Xs[i_N + j] = val;\n" +
                "                    X[(j / S) >> 0][(j % S) * N + i] = val;\n" +
                "                }\n" +
                "            }}");
    }

    @Override
    public void initializeTest(int iteration) {
        this.initialize_random_symmetric_matrix.execute(this.A, this.S, config.size);
    }

    @Override
    public void resetIteration(int iteration) {
        // Reset result
        for (int i = 0; i < config.size; i++)
            this.x.setArrayElement(i, 1.0 / config.size);
        this.t1.setArrayElement(0, 0.0);
        this.t2.setArrayElement(0, 0.0);
    }

    @Override
    public void runTest(int iteration) {
        long start_comp = System.nanoTime();
        long end;

        // Initialization phase
        // precondition: A += I * np.eps;
        for (int i = 0; i < this.P; i++) {
            this.precondition_kernel.execute(config.numBlocks, config.blockSize1D).
                    execute(this.A[i], config.size, Math.min(this.S, config.size - i * this.S), i * this.S);
        }

        // r = b - A * x
        for (int i = 0; i < this.P; i++) {
            this.mmul_axpy_kernel.execute(config.numBlocks, config.blockSize1D).
                    execute(this.A[i], this.x, this.b, -1, this.r, this.S, config.size, i * this.S);
        }

        // p = r
        this.copy_kernel.execute(config.numBlocks, config.blockSize1D).
                execute(this.p, this.r, config.size);

        // t1 = r^t * r
        this.l2_norm_kernel.execute(config.numBlocks, config.blockSize1D).
                execute(this.r, this.t1, config.size);

        for (int curr_iter = 0; curr_iter < this.ITER; curr_iter++) {
            // t2 = p^t * A * p
            for (int i = 0; i < this.P; i++) {
                this.mmul_kernel.execute(config.numBlocks, config.blockSize1D).
                        execute(this.A[i], this.p, this.y, this.S, config.size, i * this.S);
            }
            this.dp_kernel.execute(config.numBlocks, config.blockSize1D).
                    execute(this.p, this.y, this.t2, config.size);

            float alpha = this.t1.getArrayElement(0).asFloat() / this.t2.getArrayElement(0).asFloat();
            float old_r_norm_squared = this.t1.getArrayElement(0).asFloat();
            this.t1.setArrayElement(0, 0);
            this.t2.setArrayElement(0, 0);

            // Update x: x = x + alpha * p
            this.saxpy_kernel.execute(config.numBlocks, config.blockSize1D).
                    execute(this.x, this.x, this.p, alpha, config.size);

            // r = r - alpha * y
            this.saxpy_kernel.execute(config.numBlocks, config.blockSize1D).
                    execute(this.r, this.r, this.y, -1 * alpha, config.size);

            // t1 = r^t * r
            this.l2_norm_kernel.execute(config.numBlocks, config.blockSize1D).
                    execute(this.r, this.t1, config.size);

            float beta = this.t1.getArrayElement(0).asFloat() / old_r_norm_squared;

            this.saxpy_kernel.execute(config.numBlocks, config.blockSize1D).
                    execute(this.p, this.r, this.p, beta, config.size);
        }

        // Add final sync step
        float tmp = x.getArrayElement(0).asFloat();
        end = System.nanoTime();

        benchmarkResults.setCurrentComputationSec((end - start_comp) / 1000000000F);

        // Compute GPU result
        for (int i = 0; i < this.P; i++) {
            this.mmul_axpy_kernel.execute(config.numBlocks, config.blockSize1D).
                    execute(this.A[i], this.x, this.b, -1, this.y, Math.min(this.S, config.size - i * this.S), config.size, i * this.S);
        }

        float sum = 0;
        for (int i = 0; i < 10; i++)
            sum += this.y.getArrayElement(i).asFloat();

        benchmarkResults.setCurrentGpuResult(0);
    }

    @Override
    public void cpuValidation() {
        float[][] A_cpu = new float[config.size][config.size];
        float[] b_cpu = new float[config.size];
        float[] x_cpu_1 = new float[config.size];
        float[] x_cpu = new float[config.size];
        float[] r_cpu = new float[config.size];
        float[] p_cpu = new float[config.size];
        float[] y_cpu = new float[config.size];
        float[] tmp;
        float t1_cpu = 0;
        float t2_cpu = 0;
        float alpha_cpu;
        float beta_cpu;
        float t1_old_cpu;

        for (int i = 0; i < config.size; i++) x_cpu_1[i] = 0;

        int p_counter;
        for (int i = 0; i < config.size; i++) {
            p_counter = Math.floorDiv(i, this.S);
            for (int j = 0; j < config.size; j++)
                A_cpu[i][j] = this.A[p_counter].getArrayElement((i % this.S) * config.size + j).asFloat();
        }

//        System.out.println("Matrix test A-CPU");
//        System.out.println("Matrix A-CPU -> rowSize: " + A_cpu.length + "; colSize: " + A_cpu[0].length);
//        for (int r=0; r<config.size; r++) {
//            System.out.print('|');
//            for (int c=0; c<config.size; c++) {
//                System.out.print(A_cpu[r][c] + "\t| ");
//            }
//            System.out.print('\n');
//        }

        Random rd = new Random();
        for (int i = 0; i < config.size; i++) b_cpu[i] = rd.nextFloat();

        for (int i = 0; i < config.size; i++) x_cpu[i] = 1;

        tmp = matrixMult(A_cpu, x_cpu);
        for (int i = 0; i < config.size; i++) r_cpu[i] = b_cpu[i] - tmp[i];

        for (int i = 0; i < config.size; i++) p_cpu[i] = r_cpu[i];

        for (int i = 0; i < config.size; i++) t1_cpu += (r_cpu[i] * r_cpu[i]);

        // Main iteration
        for (int i = 0; i < ITER; i++) {
            y_cpu = matrixMult(A_cpu, p_cpu);

            for (int j = 0; j < config.size; j++) t2_cpu += (p_cpu[j] * y_cpu[j]);

            alpha_cpu = t1_cpu / t2_cpu;
            t1_old_cpu = t1_cpu;
            for (int j = 0; j < config.size; j++){
                x_cpu[j] += alpha_cpu * p_cpu[j];
                r_cpu[j] -= alpha_cpu * y_cpu[j];
            }

            for (int j = 0; j < config.size; j++) t1_cpu += (r_cpu[j] * r_cpu[j]);

            beta_cpu = t1_cpu / t1_old_cpu;

            for (int j = 0; j < config.size; j++) p_cpu[j] = r_cpu[j] + beta_cpu * p_cpu[j];
        }

//        System.out.println(" CPU - y pre sum ");
//        for (int i=0; i < config.size; i++) {System.out.print(y_cpu[i] + " % ");}
//        System.out.print("\n");
        float sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += y_cpu[i];
        }

        benchmarkResults.setCurrentCpuResult(sum);
        assertEquals(benchmarkResults.currentCpuResult(), benchmarkResults.currentGpuResult(), 1e-3);
    }

    private float[] matrixMult(float[][] a, float[] b) {
        float[] res = new float[a.length];
        float tempSum;

        for (int r = 0; r < a.length; r++) {
            tempSum = 0;
            for (int k = 0; k < b.length; k++) {
                tempSum += a[r][k] * b[k];
            }
            res[r] = tempSum;
        }
        return res;
    }
}