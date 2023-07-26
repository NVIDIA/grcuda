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
import static org.junit.Assert.assertEquals;


public class B1M extends Benchmark {

    private static final String SQUARE_KERNEL = "" +
            "extern \"C\" __global__ void square(float* x, float* y, int n) { \n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        y[i] = x[i] * x[i];\n" +
            "    }\n" +
            "}\n";

    private static final String REDUCE_KERNEL = "" +
            "// From https://devblogs.nvidia.com/faster-parallel-reductions-kepler/\n" +
            "\n" + "__inline__ __device__ float warp_reduce(float val) {\n" +
            "    int warp_size = 32;\n" + "    for (int offset = warp_size / 2; offset > 0; offset /= 2)\n" +
            "        val += __shfl_down_sync(0xFFFFFFFF, val, offset);\n" +
            "    return val;\n" + "}\n" + "\n" + "__global__ void reduce(float *x, float *y, float* z, int N) {\n" +
            "    int warp_size = 32;\n" + "    float sum = float(0);\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {\n" +
            "        sum += x[i] - y[i];\n" + "    }\n" +
            "    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;\n" +
            "    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster\n" +
            "        atomicAdd(z, sum); // The first thread in the warp updates the output;\n" +
            "}";

    private Value squareKernelFunction;
    private Value reduceKernelFunction;
    private Value initialize;
    private ArrayList<Value> x, x1, y, y1, res;
    //private Value initialize;
    double res_tot=0;
    private int partitionSize;
    private final int P = 16;

    public B1M(BenchmarkConfig currentConfig) {
        super(currentConfig);
    }

    @Override
    public void initializeTest(int iteration) {
        assert (!config.randomInit);
        for(int i=0; i<P; i++){
            initialize.execute(x.get(i), i, config.size, 1.0f);
            initialize.execute(y.get(i), i, config.size, 2.0f);

        }
    }

    private void initializeWithJava(Value x, int i, int N, float a){
        long index;
        for(int j = 0; j<x.getArraySize(); j++){
            index = i * x.getArraySize() + j;
            if(index < N ){
                x.setArrayElement(j, a / (index+1));
            }
        }
    }

    @Override
    public void allocateTest(int iteration) {
        // Compute the partition size
        partitionSize = (config.size + P -1) / P;

        // Alloc arrays
        x = new ArrayList<>();
        x1 = new ArrayList<>();
        y = new ArrayList<>();
        y1 = new ArrayList<>();
        res = new ArrayList<>();

        for(int i=0; i<P; i++){
            x.add(requestArray("float", partitionSize));
            x1.add(requestArray("float", partitionSize));
            y.add(requestArray("float", partitionSize));
            y1.add(requestArray("float", partitionSize));
            res.add(requestArray("float", 1));
        }

        // Context initialization
        Value buildKernel = context.eval("grcuda", "buildkernel");

        // Build the kernels;
        squareKernelFunction = buildKernel.execute(SQUARE_KERNEL, "square", "pointer, pointer, sint32");
        reduceKernelFunction = buildKernel.execute(REDUCE_KERNEL, "reduce", "pointer, pointer, pointer, sint32");

        initialize = context.eval("js", "(x, i, N, a) => { for (let j = 0; j < x.length; j++) { let index = i * x.length + j; if (index < N) {x[j] = a / (index + 1); }}}");

    }

    @Override
    public void resetIteration(int iteration) {
        for(int i=0; i<P; i++){
            initialize.execute(x.get(i), i, config.size, 1);
            initialize.execute(y.get(i), i, config.size, 2);
            //initializeWithJava(x.get(i), i, config.size, 1.0f);
            //initializeWithJava(y.get(i), i, config.size, 2.0f);
            res.get(i).setArrayElement(0, 0.0f);
        }
        res_tot = 0;
    }

    @Override
    public void runTest(int iteration) {
        long start = System.nanoTime();

        for(int i=0; i<P; i++){
            // A, B. Call the kernel. The 2 computations are independent, and can be done in parallel;
            squareKernelFunction.execute(config.numBlocks, config.blockSize1D).execute(x.get(i), x1.get(i), partitionSize);
            squareKernelFunction.execute(config.numBlocks, config.blockSize1D).execute(y.get(i), y1.get(i), partitionSize);
            // C. Compute the sum of the result;
            reduceKernelFunction.execute(config.numBlocks, config.blockSize1D).execute(x1.get(i), y1.get(i), res.get(i), partitionSize);
        }

        for(int i=0; i<P; i++){
            float val = res.get(i).getArrayElement(0).asFloat();
            if(!Float.isNaN(val))
                res_tot += val;
        }
        long end = System.nanoTime();

        // Sync step to measure the real computation time
        benchmarkResults.setCurrentGpuResult(res_tot);
        benchmarkResults.setCurrentComputationSec((end-start)/1000000000F);
    }


    @Override
    public void cpuValidation() {
        assert (!config.randomInit);

        float[] xHost = new float[config.size];
        float[] yHost = new float[config.size];

        for (int i = 0; i < config.size; i++) {
            xHost[i] = 1.0f / (i + 1);
            yHost[i] = 2.0f / (i + 1);
        }

        for (int i = 0; i < config.size; i++) {
            xHost[i] = xHost[i] * xHost[i];
            yHost[i]=  yHost[i] * yHost[i];
            xHost[i] -= yHost[i];
        }

        double acc = 0.0f;

        for (int i = 0; i < config.size; i++) {
            acc += xHost[i];
        }

        benchmarkResults.setCurrentCpuResult(acc);

        assertEquals(benchmarkResults.currentCpuResult(), benchmarkResults.currentGpuResult(), 1e-3);

    }

}
