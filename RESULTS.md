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

| Threads | Size | Sync time (s) | DAG (s) | DAG Speedup |
|-----|-----|-----|-----|-----|
|  32  |  2000000   |  0.0020 | 0.0016 |   1.25x  |  
|      |  20000000   |  0.0125  |   0.0063  |  1.98x  |  
|  1024  |   2000000  |  0.0013  | 0.0013   | 1x    | 
|     |   20000000  |  0.0074 | 0.0037   |  2x | 

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

| Threads | Size | Sync time (s) | DAG (s) | DAG Speedup |
|-----|-----|-----|-----|-----|
|  32  | 50000    |  0.100   |  0.068   |  1.47x  | 
|      |  500000   | 1.160   |  0.91  |  1.27x   |  
|  1024   | 50000    |   0.122  |  0.103   |   1.18x  | 
|     |   500000  |  2.11  |  2.02   |   1.04x  | 

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


| Threads | Vertices | Degree | Sync time (s) | DAG (s) | DAG Speedup |
|-----|-----|-----|-----|-----|-----|
|  32  | 100000  | 10       |  0.020   |  0.011   |  1.81x   | 
|      |   | 100            | 0.088|  0.042         |  2.09x   |
|    |       1000000 | 10   |  0.224   |   0.196  |   1.14x   | 
|  1024   |  100000 | 10    |   0.016  |  0.012   |  1.33x   |
|      |   | 100            | 0.191 |      0.174      |   1.09x |
|     |   1000000   | 10    | 0.232  |  0.212  |  1.09x   | 

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

| Threads | Size | Sync time (s) | DAG (s) | DAG Speedup |
|-----|-----|-----|-----|-----|
|  32  |  2000     |  0.0241   |  0.0128   |  1.88x    | 
|      |  4000     | 0.0890   | 0.0760   |  1.17x   |  
|  1024  |   2000  |   0.056  |  0.040   |  1.4x  | 
|     |   4000     |   0.179  |  0.169    |  1.05x   | 


