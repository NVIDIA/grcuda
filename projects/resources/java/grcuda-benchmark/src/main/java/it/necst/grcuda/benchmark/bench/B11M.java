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

import static org.junit.Assert.assertEquals;

public class B11M extends Benchmark {
    /*
     *  Dense matrix-vector multiplication, partitioning the matrix in blocks of rows;
     */

    private static final String MATRIX_VECTOR_MULT_KERNEL = "" +
            "extern \"C\" __global__ void matrix_vector_mult_1(const float* x, const float* y, float* z, int n, int m, int z_offset) {\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        float sum = 0;\n" +
            "        for (int j = 0; j < m; j++) {                \n" +
            "            sum += x[i * m + j] * y[j];\n" +
            "        }\n" +
            "        z[z_offset + i] = sum;\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "extern \"C\" __global__ void matrix_vector_mult_2(const float* x, const float* y, float* z, int n, int m) {\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        float sum = 0;\n" +
            "        for (int j = 0; j < m; j++) {                \n" +
            "            sum += x[i * m + j] * y[j];\n" +
            "        }\n" +
            "        z[i] = sum;\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "extern \"C\" __global__ void copy(const float *x, float *y, int n, int offset) {\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        y[i + offset] = x[i];\n" +
            "    }\n" +
            "}";

    private Value matrix_vector_mult_kernel, copy_kernel, initialize;
    private Value[] x, z;
    private Value y, z_out;
    private float[][] x_cpu_matrix;
    private float[] x_cpu_array, y_cpu;
    private int N, M, P, S;

    public B11M(BenchmarkConfig currentConfig) {
        super(currentConfig);

        // Square matrix of size x size
        this.N = config.size;
        this.M = config.size;

        // Use P horizontal partitions
        this.P = 16;

        // Size of partitions
        this.S = Math.floorDiv(this.N + this.P - 1, this.P);

        // Full matrix
        this.x_cpu_array = null;
        this.x_cpu_matrix = null;
        // Dense vector
        this.y_cpu = null;

        // The GPU matrix is stored using P arrays
        this.x = new Value[this.P];
        for (int i = 0; i < this.P; i++) {
            this.x[i] = null;
        }
        // Dense vector
        this.y = null;
        // Result
        // this.z = null;
        this.z = new Value[this.P];
        for (int i = 0; i < this.P; i++) {
            this.z[i] = null;
        }
        this.z_out = null;

        this.matrix_vector_mult_kernel = null;
    }

    @Override
    public void allocateTest(int iteration) {
        this.N = config.size;
        this.M = config.size;
        this.S = Math.floorDiv(this.N + this.P - 1, this.P);

        // Allocate vectors
        for (int i = 0; i < this.P; i++) {
            this.x[i] = requestArray("float", this.S * this.M);
        }
        this.y = requestArray("float", this.M);
        // this.z = requestArray("float", this.N);
        for (int i = 0; i < this.P; i++) {
            this.z[i] = requestArray("float", this.S);
        }
        this.z_out = requestArray("float", this.N);

        // Build the kernels;
        Value buildKernel = context.eval("grcuda", "buildkernel");
        // this.matrix_vector_mult_kernel = buildKernel.execute(MATRIX_VECTOR_MULT_KERNEL, "matrix_vector_mult_2", "const pointer, const pointer, pointer, sint32, sint32, sint32")
        this.matrix_vector_mult_kernel = buildKernel.execute(MATRIX_VECTOR_MULT_KERNEL, "matrix_vector_mult_2", "const pointer, const pointer, pointer, sint32, sint32");
        this.copy_kernel = buildKernel.execute(MATRIX_VECTOR_MULT_KERNEL, "copy", "const pointer, pointer, sint32, sint32");
        this.initialize = context.eval("js", "x => { for (let i = 0; i < x.length; i++) { x[i] = i / x.length }}");
    }

    @Override
    public void initializeTest(int iteration) {
        assert (!config.randomInit); // randomInit not supported yet
    }

    @Override
    public void resetIteration(int iteration) {
        // Reset result

        for (int i = 0; i < this.P; i++) this.initialize.execute(this.x[i]);
        for (int i = 0; i < this.M; i++) {
            this.y.setArrayElement(i, (float)(i) / this.M);
        }
    }

    @Override
    public void runTest(int iteration) {
        long start = System.nanoTime();

        // Compute all partitions
        for (int p = 0; p < this.P; p++) {
            // this.matrix_vector_mult_kernel.execute(config.numBlocks, config.blockSize1D)
            //         .execute(this.x[p], this.y, this.z, Math.min(this.S, this.N - p * this.S), this.M, p * this.S);
            this.matrix_vector_mult_kernel.execute(config.numBlocks, config.blockSize1D).
                    execute(this.x[p], this.y, this.z[p], Math.min(this.S, this.N - p * this.S), this.M);
        }

        // Aggregate results
        for (int p = 0; p < this.P; p++) {
            this.copy_kernel.execute(config.numBlocks, config.blockSize1D).
                    execute(this.z[p], this.z_out, Math.min(this.S, this.N - p * this.S), p * this.S);
        }

        // Add a final sync step to measure the real computation time  Math.sum(self.z_out[:10])
        float sum = 0;
        for (int i = 0; i < 10; i++) sum += this.z_out.getArrayElement(i).asFloat();

        long end = System.nanoTime();

        benchmarkResults.setCurrentGpuResult(sum);
        benchmarkResults.setCurrentComputationSec((end-start)/1000000000F);

    }

    @Override
    public void cpuValidation() {
        float[] z_cpu;
        float sum;

        x_cpu_array = new float[this.N * this.M];
        x_cpu_matrix = new float[this.N][this.M];
        y_cpu = new float[this.M];

        for (int i = 0; i < this.N * this.M; i++) x_cpu_array[i] = 0.0F;
        for (int i = 0; i < this.M; i++) y_cpu[i] = this.y.getArrayElement(i).asFloat();

        for (int i = 0; i < this.P; i++) {
            for (int j = 0; j < this.S * this.M; j++) {
                if (i * this.S * this.M + j < x_cpu_array.length) {
                    x_cpu_array[i * this.S * this.M + j] = (float) (j) / (this.S * this.M);
                }
            }
        }
        for (int r = 0; r < this.N; r++)
            for (int c = 0; c < this.M; c++) {
                x_cpu_matrix[r][c] = x_cpu_array[r * this.M + c];
            }
        z_cpu = matrixMult(x_cpu_matrix, y_cpu);

        sum = 0;
        for (int i = 0; i < 10; i++) sum += z_cpu[i];
        benchmarkResults.setCurrentCpuResult(sum);

        // Compare GPU and CPU results
        assertEquals(benchmarkResults.currentGpuResult(), sum, 1e-4);
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