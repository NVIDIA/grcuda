# Benchmark Results

This document contains some performance results obtained running Graalpython benchmarks in `projects/resources/python/benchmark/bench`.
 We compare the performance achieved with DAG scheduling compared to a synchronous GrCUDA execution, and to a native CUDA application when possible.
 
## Setup

* **GPU**: Nvidia GTX 960, 2 GB
* **CPU**: Intel i7-6700 @ 3.40GHz, 8 threads
* **DRAM**: 32 GB, DDR4
* Execution time measures the total amount of time spent by GPU execution, from the first kernel scheduling until all GPU kernels have finished executing
* Each benchmark is executed for **30 iterations**, and the average time skips the first 3 to allow the performance of GraalVM to stabilize 
* Parameters that have been kept fixed (such as the number of blocks for kernels whose number of blocks is not a function of data-size) have been optimized to provide the best performance for synchronous execution,
and provide a more realistic performance comparison.

## Results

* **Sync time** is the baseline, it measures synchronous GPU kernel scheduling. In this case, dependencies between kernels are not computed, making GrCUDA overheads even smaller.
* **DAG** is the computation time when using GrCUDA DAG kernel scheduling, performing transparent GPU resource-sharing.

* The field **Threads** is the number of threads for each block, in CUDA; this number ranges from 32 to 1024. A higher number implies bigger blocks, and possibly fewer concurrent blocks running in parallel.
 Blocks can also be 2-dimensional: we left 2D blocks with size 8x8 as bigger blocks always resulted in strictly longer execution times, for all scheduling policies.
* The field **Size** is the number of elements in the input. 
Depending on the benchmark it could be the size of a vector, the number of rows in a square matrix, the number of vertices of a graph; more information are provided for each benchmark 


### Vector Squares (bench_1)

Compute the sum of difference of squares of 2 vectors, using multiple GrCUDA kernels. It's a fairly artificial benchmark that measures a simple case of parallelism.
Most of the execution time is spent in the reduction computation, limiting the amount of parallelism available, especially on large input data.
Speedups are achievable by overlapping data-transfer and computations, although the data-transfer takes about 4x-5x longer than the square computation, limiting the maximum achievable speedup.

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

### Black & Scholes (bench_5)

Compute the Black & Scholes equation for European call options, for 10 different underlying types of stocks, and for each stock a vector of prices at time 0. 
The main computation is taken from Nvidia's CUDA code samples ([link](http://mercury.pr.erau.edu/~siewerts/extra/code/digital-media/CUDA/cuda_work/samples/4_Finance/BlackScholes/BlackScholes_kernel.cuh)), 
and adapted to use double precision arithmetic to create a more computationally intensive kernel.
The idea of this benchmark is to simulate a streaming computation in which data-transfer and computation of multiple kernels can be overlapped efficiently,
without data-dependencies between kernels.
To the contrary of `bench_1`, computation, and not data transfer, is the main limiting factor for parallel execution.

Structure of the computation:

```
BS(x[1]) -> ... -> BS(x[10])
```

### Machine Learning Ensemble (bench_6)

Compute an ensemble of Categorical Naive Bayes and Ridge Regression classifiers.
Predictions are aggregated averaging the class scores after softmax normalization.
The computation is done on mock data and parameters, but is conceptually identical to a real ML pipeline.
In the DAG below, input arguments that are not involved in the computation of dependencies are omitted.

The size of the benchmark is the number of rows in the matrix (each representing a document with 1000 features). Predictions are done by choosing among 5 classes.
The Ridge Regression classifier takes about 2x the time of the Categorical Naive Bayes classifier.

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

The input graph has **size** vertices, degree 3 and uniform distribution. Each execution of this algorithm is composed of 5 iterations.
The number of blocks is kept constant at 32, as higher block count resulted in worse overall performance.

As the benchmark is composed of 2 independent branches, the **maximum theoretical speedup is 2x**, although realistic speedup will be lower and mostly achieved through transfer-computation overlapping.

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

![Speedup w.r.t. serial, summary](https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_08_052/speedup_baseline_1_row_2020_08_052.png)

In general, **DAG scheduling allows better GPU resource usage**: in many benchmarks (such as `bench_1` and `bench_5`) we observe how overlapping data-transfer and computation enables faster total execution time.
 In other cases (`bench_6` and `bench_7`) speedups are also provided by computation overlap. DAG schedling provides **speedups from 10% to 50%**, while being completely transparent to the programmer.

**Speedup stays constant with data-set size. even for large data-sets** (enough to saturate the GPU memory): even if kernels fill the GPU resources,
 it is still possible to achieve a large degree of parallelism by overlapping data transfer with execution of different kernels, or possibly by overlapping execution of kernels with different bottlenecks.

**DAG scheduling is never worse than serial scheduling**, meaning that users do not have to think about which scheduling policy would be better for them, but can always leverage DAG scheduling.

As rule of thumb, **DAG scheduling is more robust to kernel configuration**: in many cases, using `block_size=32` results in higher speedup, but similar absolute execution time compared to larger block size.
 In the case of serial synchronous scheduling, using small blocks results in underutilization of the GPU resources, while using DAG scheduling provides better resource utilization by scheduling multiple kernels in parallel.
 This is an extremely useful advantage of DAG scheduling, as it means that programmers have to spend less time profiling their code to find the optimal kernel configuration.

![Speedup w.r.t. serial, extended](https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_08_052/speedup_baseline_2020_08_052.png)

Performance of serial and DAG GrCUDA scheduling has been compared to the same benchmarks implemented directly in C++ and CUDA. The experimental setup and the kernels are exactly the same. In the case of CUDA asynchronous kernel execution, dependencies and synchronization points have been computed by hand, instead of automatically. This provides a **comparison of how the overhead introduced by GrCUDA impacts the total execution time** compared to lower-level kernel scheduling.

In all cases, the gap between CUDA and GrCUDA is minimal, and converges to 0 as the data-set size increases. It can be safely stated that in any realistic computation using serial GrCUDA scheduling will not decrease performance. 
The only situation with a visible difference is `bench_8`, when processing very small images (`800x800` pixels), as the computation lasts only for a couple of milliseconds.

![Relative exec. time w.r.t. CUDA, summary](https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_08_052/speedup_baseline_grcuda_cuda_compact_2020_08_052.png)

![Relative exec. time w.r.t. CUDA, extended](https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_08_052/speedup_baseline_grcuda_cuda_2020_08_052.png)

These results are further reaffirmed by looking at the execution time distributions (for example, using the largest data-sets and `block_size=256`, to evaluate an average case).
 In a couple of cases (e.g `bench_1`) it can be seen how the CUDA implementation might be sligthly faster (around 5%), although the same is true for GrCUDA in `bench_8`, while other benchmarks show very similar distributions.

<img src="https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_07_15/speedup_baseline_grcuda_cuda_ridgeplot_2020_07_15.png" width="600">

By measuring the execution time of each phase of a benchmark, it is possible to estimate the fastest theoretical time of the benchmark.
 This theoretical time assumes a *GPU* with infinite resources and infinite host-device capacity, so that it's always possible to transfer data without having to wait for the current transfer to end 
 (transfer bandwidth is still limited by PCIe 3.0 speed, at 16 GB/s bidirectional). We compute theoretical speed for each benchmark by looking at the dependencies between computations, 
 and understanding which computations, given infinite resources, could be scheduled in parallel without any slow-down.

DAG scheduling always provides execution times closer to the theoretical optimum than serial scheduling, as expected.
 In many benchmarks we are very close to the theoretical optimum, showing how DAG scheduling provides better GPU resource utilization and transfer-computation overlap.
  Not surprisingly, the only case where results are significantly lower than the optimum is `bench_5`, as it is composed of many independent computations that could in principle run in parallel: this is clearly not possible in practice, although DAG scheduling proves once again to be better than serial execution.

![Relative exec. time w.r.t. theoretical minimum time](https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_08_052/speedup_theoretical_time_compact_2020_08_052.png)

### Performance analysis

![Overlap amount](https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_08_052/overlap_2020_08_052.png)

For each benchmark, we can measure how much **overlap** is present in the computation. We measure 4 different types of overlap:

1. **CT, computation w.r.t transfer**: percentage of GPU kernel computation that overlaps with any data transfer (host-to-device or viceversa)
2. **TC, transfer w.r.t computation**: percentage of data transfer that overlaps with (one or more) GPU kernel computation(s)
3. **CC, computation w.r.t computation**: percentage of GPU kernel computation that overlaps with any other GPU kernel computation
4. **TOT, any type of overlap**: here we consider any type of overlap between data-transfer and/or computations.
 Note that if a computation/data-transfer overlaps more than one computation/data-transfer, the overlap is counted only once (we consider the union of the overlap intervals)
 
Measures are taken for the largest data-size in the evaluation (for each benchmark), for the block size that results in higher speedup,
 to obtain a clearer understanding of what type of overlap is providing the speedup.
 In general, the **TOT** overlap is a good proxy of the achieved speedup, although it is sometimes inflated by high **CC** overlap: 
 in fact, overlapping computations does not always translates to faster execution, especially if kernels are large enough (in terms of threads/blocks) to fill the GPU processors on their own.
 We observe how in `b1` the speedup comes exclusively from the overlap of transfer and computation, while in `b8` the speedup is caused by the overlap of kernels which,
  if executed serially, do not fill the GPU resources. Very different values of **CT** and **TC** (as in `b5`) indicate that, although the data-transfer is completely overlapped to GPU computations, 
  the computation lasts much longer than the data-transfer, and part of the computation cannot be overlapped. 
  In all likelihood, a more optimized kernel computation would result in higher **CT** overlap, and better speedup.
  
<img src="https://github.com/AlbertoParravicini/grcuda/blob/execution-model-sync/data/plots/2020_08_052/memory_throughput_2020_08_052.png">

Using `nvprof` we measure the total amount of bytes read/written by each kernel, and analyse how the GPU memory throughput is affected by space-sharing. 
Note that `nvprof` affects the kernel execution and limits the execution of concurrent kernels due to the high overhead introduced by collecting memory access metrics for each kernel.
Instead, we measure the execution times obtained without metric collection (so that `nvprof` influence over the execution times is minimal) 
and combine them with memory access metrics collected in a separate run. 
The assumption here is the total amount of memory accesses is not significantly impacted by `nvprof` profiling, and this evaluation is still useful to obtain performance insights.

Indeed, we see that for kernels that contain computation overlap (e.g. `b6` and `b8`) the increase in memory throughput is significant, and in-line with the total speedup observed for these benchmarks.
As expected, `b1` does not have any increase in memory throughput, as its speedup comes exclusively from transfer overlap.
 Similarly, `b5` shows a small memory throughput increase, as the benchmark main bottleneck by the arithmetic intensity, and a significant part of the speedup comes from transfer overlapping.
 
Other metrics, such as L2 cache read/write throughput, and IPC (computed assuming that the GPU runs at the maximum frequency), show identical speedups; 
benchmarks that operates on matrices (`b6` and `b8`) make heavier use of the cache, while the sparse matrices in `b6` results in very low IPC, due to the high number of random memory accesses.
