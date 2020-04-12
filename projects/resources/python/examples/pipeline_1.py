import polyglot
import time
import math

NUM_THREADS_PER_BLOCK = 128

SQUARE_KERNEL = """
    extern "C" __global__ void square(float* x, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            x[idx] = x[idx] * x[idx];
        }
    }
    """

DIFF_KERNEL = """
    extern "C" __global__ void diff(float* x, float* y, float* z, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            z[idx] = x[idx] - y[idx];
        }
    }
    """

REDUCE_KERNEL = """
    extern "C" __global__ void reduce(float *x, float *res, int n) {
        __shared__ float cache[%d];
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            cache[threadIdx.x] = x[i];
        }
        __syncthreads();
    
        // Perform tree reduction;
        i = %d / 2;
        while (i > 0) {
            if (threadIdx.x < i) {
                cache[threadIdx.x] += cache[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }
        if (threadIdx.x == 0) {
            atomicAdd(res, cache[0]);
        }
    }
    """ % (NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK)

# Compute the sum of difference of squares of 2 vectors, using multiple GrCUDA kernels.
# Structure of the computation:
#   A: x^2 ──┐
#            ├─> C: z=x-y ──> D: sum(z)
#   B: x^2 ──┘
if __name__ == "__main__":
    N = 100000
    NUM_BLOCKS = (N + NUM_THREADS_PER_BLOCK - 1) // NUM_THREADS_PER_BLOCK

    start_tot = time.time()
    time_cumulative = 0

    # Allocate 2 vectors;
    start = time.time()
    x = polyglot.eval(language="grcuda", string=f"float[{N}]")
    y = polyglot.eval(language="grcuda", string=f"float[{N}]")

    # Allocate a support vector;
    z = polyglot.eval(language="grcuda", string=f"float[{N}]")
    res = polyglot.eval(language="grcuda", string=f"float[1]")
    end = time.time()
    time_cumulative += end - start
    print(f"time to allocate arrays: {end - start:.4f} sec")

    # Fill the 2 vectors;
    start = time.time()
    for i in range(N):
        x[i] = 1 / (i + 1)
        y[i] = 2 / (i + 1)
    res[0] = 0
    end = time.time()
    time_cumulative += end - start
    print(f"time to fill arrays: {end - start:.4f} sec")

    # A. B. Compute the squares of each vector;

    # First, build the kernel;
    build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
    square_kernel = build_kernel(SQUARE_KERNEL, "square", "pointer, sint32")

    # Call the kernel. The 2 computations are independent, and can be done in parallel;
    start = time.time()
    square_kernel(NUM_BLOCKS, NUM_THREADS_PER_BLOCK)(x, N)
    square_kernel(NUM_BLOCKS, NUM_THREADS_PER_BLOCK)(y, N)
    end = time.time()
    time_cumulative += end - start
    print(f"square, time: {end - start:.4f} sec")

    # C. Compute the difference of the 2 vectors. This must be done after the 2 previous computations;
    start = time.time()
    diff_kernel = build_kernel(DIFF_KERNEL, "diff", "pointer, pointer, pointer, sint32")
    diff_kernel(NUM_BLOCKS, NUM_THREADS_PER_BLOCK)(x, y, z, N)
    end = time.time()
    time_cumulative += end - start
    print(f"diff, time: {end - start:.4f} sec")

    # D. Compute the sum of the result;
    start = time.time()
    reduce_kernel = build_kernel(REDUCE_KERNEL, "reduce", "pointer, pointer, sint32")
    reduce_kernel(NUM_BLOCKS, NUM_THREADS_PER_BLOCK)(z, res, N)
    end = time.time()
    time_cumulative += end - start
    print(f"reduce, time: {end - start:.4f} sec")
    print(f"overheads, time: {end - start_tot - time_cumulative:.4f} sec")
    print(f"total time: {end - start_tot:.4f} sec")

    result = res[0]
    print(f"result={result:.4f}")
