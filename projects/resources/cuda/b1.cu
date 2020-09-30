#include "b1.cuh"

//////////////////////////////
//////////////////////////////

__global__ void square(const float *x, float *y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // float tmp = x[i];
        // float sum = 0;
        // for (int j = 0; j < 4; j++) {
        //     sum += tmp + j;
        // }

        y[i] = x[i] * x[i];  // tmp + tmp * tmp / 2 + tmp * tmp * tmp / 6;
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// __device__ float atomicAddDouble(float* address, float val) {
//     unsigned long long int* address_as_ull = (unsigned long long int*) address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed, __float_as_longlong(val + __longlong_as_float(assumed)));
//     } while (assumed != old);
//     return __longlong_as_float(old);
// }

__global__ void reduce(const float *x, const float *y, float *z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] - y[i];
    }
    sum = warp_reduce(sum);                    // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum);                     // The first thread in the warp updates the output;
}

void Benchmark1::prefetch(cudaStream_t &s1, cudaStream_t &s2) {
    if (pascalGpu) {
        cudaMemPrefetchAsync(x, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(y, sizeof(float) * N, 0, s2);
        cudaMemPrefetchAsync(x1, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(y1, sizeof(float) * N, 0, s2);
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark1::alloc() {
    err = cudaMallocManaged(&x, sizeof(float) * N);
    err = cudaMallocManaged(&y, sizeof(float) * N);
    err = cudaMallocManaged(&x1, sizeof(float) * N);
    err = cudaMallocManaged(&y1, sizeof(float) * N);
    err = cudaMallocManaged(&res, sizeof(float));

    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);
}

void Benchmark1::init() {
    for (int i = 0; i < N; i++) {
        x[i] = 1.0 / (i + 1);
        y[i] = 2.0 / (i + 1);
    }
}

void Benchmark1::reset() {
    for (int i = 0; i < N; i++) {
        x[i] = 1.0 / (i + 1);
        y[i] = 2.0 / (i + 1);
    }
    res[0] = 0.0;
}

void Benchmark1::execute_sync(int iter) {
    if (do_prefetch && pascalGpu) {
        cudaMemPrefetchAsync(x, sizeof(float) * N, 0, 0);
        cudaMemPrefetchAsync(x1, sizeof(float) * N, 0, 0);
        cudaMemPrefetchAsync(y, sizeof(float) * N, 0, 0);
        cudaMemPrefetchAsync(y1, sizeof(float) * N, 0, 0);
        cudaMemPrefetchAsync(res, sizeof(float), 0, 0);
    }

    square<<<num_blocks, block_size_1d>>>(x, x1, N);
    err = cudaDeviceSynchronize();
    square<<<num_blocks, block_size_1d>>>(y, y1, N);
    err = cudaDeviceSynchronize();
    reduce<<<num_blocks, block_size_1d>>>(x1, y1, res, N);
    err = cudaDeviceSynchronize();
}

void Benchmark1::execute_async(int iter) {
    if (!pascalGpu || stream_attach) {
        cudaStreamAttachMemAsync(s1, x, sizeof(float) * N);
        cudaStreamAttachMemAsync(s1, x1, sizeof(float) * N);
        cudaStreamAttachMemAsync(s2, y, sizeof(float) * N);
        cudaStreamAttachMemAsync(s2, y1, sizeof(float) * N);
    }
    if (pascalGpu && do_prefetch) {
        cudaMemPrefetchAsync(x, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(x1, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(y, sizeof(float) * N, 0, s2);
        cudaMemPrefetchAsync(y1, sizeof(float) * N, 0, s2);
        cudaMemPrefetchAsync(res, sizeof(float), 0, s1);
    }

    square<<<num_blocks, block_size_1d, 0, s1>>>(x, x1, N);
    square<<<num_blocks, block_size_1d, 0, s2>>>(y, y1, N);

    // Stream 1 waits stream 2;
    cudaEvent_t e1;
    cudaEventCreate(&e1);
    cudaEventRecord(e1, s2);
    cudaStreamWaitEvent(s1, e1, 0);

    reduce<<<num_blocks, block_size_1d, 0, s1>>>(x1, y1, res, N);
    cudaStreamSynchronize(s1);
}

void Benchmark1::execute_cudagraph(int iter) {
    if (iter == 0) {
        cudaEvent_t ef;
        cudaEventCreate(&ef);
        cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
        cudaEventRecord(ef, s1);
        cudaStreamWaitEvent(s2, ef, 0);

        // prefetch(s1, s2);

        square<<<num_blocks, block_size_1d, 0, s1>>>(x, x1, N);
        square<<<num_blocks, block_size_1d, 0, s2>>>(y, y1, N);
        // Stream 1 waits stream 2;
        cudaEvent_t e1;
        cudaEventCreate(&e1);
        cudaEventRecord(e1, s2);
        cudaStreamWaitEvent(s1, e1, 0);
        reduce<<<num_blocks, block_size_1d, 0, s1>>>(x1, y1, res, N);

        cudaStreamEndCapture(s1, &graph);
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }
    cudaGraphLaunch(graphExec, s1);
    err = cudaStreamSynchronize(s1);
}

void Benchmark1::execute_cudagraph_manual(int iter) {
    if (iter == 0) {
        cudaGraphCreate(&graph, 0);
        void *kernel_1_args[3] = {(void *)&x, (void *)&x1, &N};
        void *kernel_2_args[3] = {(void *)&y, (void *)&y1, &N};
        void *kernel_3_args[4] = {(void *)&x1, (void *)&y1, (void *)&res, &N};

        dim3 tb(block_size_1d);
        dim3 bs(num_blocks);

        // square<<<num_blocks, block_size_1d, 0, s1>>>(x, x1, N);
        add_node(kernel_1_args, kernel_1_params, (void *)square, bs, tb, graph, &kernel_1, nodeDependencies);

        // square<<<num_blocks, block_size_1d, 0, s2>>>(y, y1, N);
        add_node(kernel_2_args, kernel_2_params, (void *)square, bs, tb, graph, &kernel_2, nodeDependencies);

        // reduce<<<num_blocks, block_size_1d, 0, s1>>>(x1, y1, res, N);
        nodeDependencies.push_back(kernel_1);
        nodeDependencies.push_back(kernel_2);
        add_node(kernel_3_args, kernel_3_params, (void *)reduce, bs, tb, graph, &kernel_3, nodeDependencies);

        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }
    cudaGraphLaunch(graphExec, s1);
    err = cudaStreamSynchronize(s1);
}

void Benchmark1::execute_cudagraph_single(int iter) {
    if (iter == 0) {
        cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);

        // prefetch(s1, s1);

        square<<<num_blocks, block_size_1d, 0, s1>>>(x, x1, N);
        square<<<num_blocks, block_size_1d, 0, s1>>>(y, y1, N);
        reduce<<<num_blocks, block_size_1d, 0, s1>>>(x1, y1, res, N);

        cudaStreamEndCapture(s1, &graph);
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }
    cudaGraphLaunch(graphExec, s1);
    err = cudaStreamSynchronize(s1);
}

std::string Benchmark1::print_result(bool short_form) {
    return std::to_string(res[0]);
}
