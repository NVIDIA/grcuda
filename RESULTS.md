# Benchmark Results

This document contains some performance results obtained running Graalpython benchmarks in `projects/resources/python/benchmark/bench`.
 We compare the performance achieved with DAG scheduling compared to a synchronous GrCUDA execution, and to a native CUDA application when possible.
 
## Setup

* **GPU**: Nvidia GTX 960, 2 GB
* **CPU**: Intel i7-6700 @ 3.40GHz, 8 threads
* **DRAM**: 32 GB, DDR4
* Execution time measures the total amount of time spent by GPU execution, from the first kernel scheduling until all GPU kernels have finished executing
* Each benchmark is executed for **30 iterations**, and the average time skips the first 3 to allow the performance of GraalVM to stabilize 

## Results

* **Sync time** is the baseline, it measures synchronous GPU kernel scheduling. In this case, dependencies between kernels are not computed, making GrCUDA overheads even smaller.
* **DAG** is the computation time when using GrCUDA DAG kernel scheduling, performing transparent GPU resource-sharing.

* The field **Threads** is the number of threads for each block, in CUDA; this number ranges from 32 to 1024. A higher number implies bigger blocks, and possibly less GPU occupation
* The field **Size** is the number of elements in the input. 
Depending on the benchmark it could be the size of a vector, the number of rows in a square matrix, the number of vertices of a graph; more information are provided for each benchmark 


### Benchmark 1 (bench_1)

Compute the sum of difference of squares of 2 vectors, using multiple GrCUDA kernels. It's a fairly artificial benchmark that measures a simple case of parallelism.
Most of the execution time is spent in the reduction computation, limiting the amount of parallelism available, especially on large input data.

Structure of the computation:

```
A: x^2 ──┐
         ├─> C: z=sum(x-y)
B: x^2 ──┘
```

<!---
| Threads | Size | Sync time (s) | DAG (s) | DAG Speedup |
|-----|-----|-----|-----|-----|
|  32  |  2000000   |  0.0020 | 0.0016 |   1.25x  |  
|      |  20000000   |  0.0125  |   0.0063  |  1.98x  |  
|  1024  |   2000000  |  0.0013  | 0.0013   | 1x    | 
|     |   20000000  |  0.0074 | 0.0037   |  2x | 
-->

### Machine Learning Ensemble (bench_6)

Compute an ensemble of Categorical Naive Bayes and Ridge Regression classifiers.
Predictions are aggregated averaging the class scores after softmax normalization.
The computation is done on mock data and parameters, but is conceptually identical to a real ML pipeline.
In the DAG below, input arguments that are not involved in the computation of dependencies are omitted.

The size of the benchmark is the number of rows in the matrix (each representing a document with 1000 features). Predictions are done by choosing among 5 classes.
The Ridge Regression classifier takes about 2x the time of the Categorical Naive Bayes classifier.
Speedups are especially noticeable for small input size, as for larger data the Ridge-Regression classifiers bottlenecks the overall computation and results in high GPU occupation.

Structure of the computation:

```
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
```

<!---
| Threads | Size | Sync time (s) | DAG (s) | DAG Speedup |
|-----|-----|-----|-----|-----|
|  32  | 50000    |  0.100   |  0.068   |  1.47x  | 
|      |  500000   | 1.160   |  0.91  |  1.27x   |  
|  1024   | 50000    |   0.122  |  0.103   |   1.18x  | 
|     |   500000  |  2.11  |  2.02   |   1.04x  | 
-->

### HITS (bench_7)

Compute the HITS algorithm on a graph. The algorithm is composed of repeated sparse matrix-vector multiplications
on a matrix and its transpose (outgoing and ingoing edges of a graph). The 2 matrix multiplications,
for each iteration, can be computed in parallel, and take most of the total computation time.

The input graph has **size** vertices, degree 10 and uniform distribution. Each execution of this algorithm is composed of 10 iterations.
Kernel computations are very fast, and the speedup increases for larger input graphs: most likely, this is the effect of having 2 SpMV running concurrently, 
which makes better use of the available memory bandwidth. The number of blocks is kept constant at 32, as higher block count resulted in worse overall performance.

As the benchmark is composed of 2 independent branches, the **maximum theoretical speedup is 2x**.

Structure of the computation (read-only parameters that do not influence the DAG are omitted):

```
 ┌─> SPMV(const H1,A2) ┬─> SUM(const A2,A_norm) ┬─> DIVIDE(A1,const A2,const A_norm) ─> CPU: A_norm=0 ─> (repeat)
 │                     └─────────┐              │
─┤                     ┌─────────│──────────────┘                                                         
 │                     │         └──────────────┐
 └─> SPMV(const A1,H2) ┴─> SUM(const H2,H_norm) ┴─> DIVIDE(H1,const H2,const H_norm) ─> CPU: H_norm=0 ─> (repeat)                       
```

<!---
| Threads | Vertices | Degree | Sync time (s) | DAG (s) | DAG Speedup |
|-----|-----|-----|-----|-----|-----|
|  32  | 100000  | 10       |  0.020   |  0.011   |  1.81x   | 
|      |   | 100            | 0.088|  0.042         |  2.09x   |
|    |       1000000 | 10   |  0.224   |   0.196  |   1.14x   | 
|  1024   |  100000 | 10    |   0.016  |  0.012   |  1.33x   |
|      |   | 100            | 0.191 |      0.174      |   1.09x |
|     |   1000000   | 10    | 0.232  |  0.212  |  1.09x   | 
-->

### Image Processing Pipeline (bench_8)

Compute an image processing pipeline in which we sharpen an image and combine it 
with copies that have been blurred at low and medium frequencies. The result is an image sharper on the edges, 
and softer everywhere else: this filter is common, for example, in portrait retouching, where a photographer desires
 to enhance the clarity of facial features while smoothing the subject' skin and the background.
 
The input is a random square single-channel image with floating-point values between 0 and 1, with side of length **size**.
The execution time is evenly distributed between Blur and Sobel filter kernels, with Sobel filter taking about 2x more time.

Structure of the computation (read-only parameters that do not influence the DAG are omitted):

```
BLUR(image,blur1) ─> SOBEL(blur1,mask1) ───────────────────────────────────────────────────────────────────────────────┐
BLUR(image,blur2) ─> SOBEL(blur2,mask2) ┬─> MAX(mask2) ──┬─> EXTEND(mask2) ──┐                                         │
                                        └─> MIN(mask2) ──┘                   │                                         │
SHARPEN(image,blur3) ─> UNSHARPEN(image,blur3,sharpened) ────────────────────┴─> COMBINE(sharpened,blur2,mask2,image2) ┴─> COMBINE(image2,blur1,mask1,image3)
```

<!---
| Threads | Size | Sync time (s) | DAG (s) | DAG Speedup |
|-----|-----|-----|-----|-----|
|  32  |  2000     |  0.0241   |  0.0128   |  1.88x    | 
|      |  4000     | 0.0890   | 0.0760   |  1.17x   |  
|  1024  |   2000  |   0.056  |  0.040   |  1.4x  | 
|     |   4000     |   0.179  |  0.169    |  1.05x   | 
-->

## Plots

![Speedup w.r.t. serial, summary](https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_07_06/speedup_baseline_1_row_2020_07_10.png)

In general, **DAG scheduling allows better GPU resource usage**: when the data-set size does not fill the GPU computational resources, DAG scheduling provides speedups close to the theoretical optimum. As expected, the speedup is less significant as the data-set size increases, as each kernel can fully use the GPU resources by itself.

**Even for large data-sets** (enough to saturate the GPU memory), **the DAG scheduling speedup stays above 1**: even if kernels fill the GPU resources, it is still possible to achieve a small degree of parallelism by overlapping data transfer with execution of different kernels, or possibly by overlapping execution of kernels with different bottlenecks.

**DAG scheduling is never worse than serial scheduling**, meaning that users do not have to think about which scheduling policy would be better for them, but can always leverage DAG scheduling.

As rule of thumb, **smaller blocks provide better speedup**: this is likely connected to the GPU architecture being better at parallelizing smaller blocks. In general, small blocks almost always provide better absolute performance, for similar reasons. Kernels in the benchmarks leverage grid-striding, meaning that the number of blocks is independent from the size of data to be processed and each thread becomes more computationally intensive as the data-size increases (instead of being constant). Currently, kernels do not use shared memory whose size depends on block size; if that was the case, bigger blocks might have an advantage.

![Speedup w.r.t. serial, extended](https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_07_06/speedup_baseline_2020_07_10.png)

Performance of serial and DAG GrCUDA scheduling has been compared to the same benchmarks implemented directly in C++ and CUDA. The experimental setup and the kernels are exactly the same. In the case of CUDA asynchronous kernel execution, dependendencies and synchronization points have been computed by hand, instead of automatically. This provides a **comparison of how the overhead introduced by GrCUDA impacts the total execution time** compared to lower-level kernel scheduling.

In the case of serial execution, CUDA is sligthly faster than GrCUDA, as expected; however, the performance difference is negligible and converges to 0 as the data-set size increases. It can be safely stated that in any realistic computation using serial GrCUDA scheduling will not decrease performance.

As for asynchronous DAG scheduling, we see how GrCUDA is actually **faster** than CUDA in most cases. It is not clear how GrCUDA is actually faster than CUDA. The only difference is that GrCUDA uses `nvrtc` (Nvidia runtime compilation library), instead of `nvcc`. Whether this provides faster kernel launches, or generates faster code, is yet to be checked. It is likely that `nvrtc` uses the same routines as `nvcc` for compilation, although kernel launches might be faster. 

![Relative exec. time w.r.t. CUDA, summary](https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_07_06/speedup_baseline_grcuda_cuda_compact_2020_07_10.png)

![Relative exec. time w.r.t. CUDA, extended](https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_07_06/speedup_baseline_grcuda_cuda_2020_07_10.png)
