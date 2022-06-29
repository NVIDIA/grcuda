// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "b7.cuh"

//////////////////////////////
//////////////////////////////

#define WARP_SIZE 32
#define THREADS_PER_VECTOR 4
#define MAX_NUM_VECTORS_PER_BLOCK (1024 / THREADS_PER_VECTOR)

/////////////////////////////
/////////////////////////////

extern "C" __global__ void spmv(const int *ptr, const int *idx, const int *val, const float *vec, float *res, int num_rows, int num_nnz) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < num_rows; n += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int i = ptr[n]; i < ptr[n + 1]; i++) {
            sum += val[i] * vec[idx[i]];
        }
        res[n] = sum;
    }
}

extern "C" __global__ void spmv2(const int *ptr, const int *idx, const int *val, const float *vec, float *res, int num_rows, int num_nnz) {
    // Thread ID in block
    int t = threadIdx.x;

    // Thread ID in warp
    int lane = t & (WARP_SIZE - 1);

    // Number of warps per block
    int warpsPerBlock = blockDim.x / WARP_SIZE;

    // One row per warp
    int row = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);

    extern __shared__ volatile float vals[];

    if (row < num_rows) {
        int rowStart = ptr[row];
        int rowEnd = ptr[row + 1];
        float sum = 0;

        // Use all threads in a warp accumulate multiplied elements
        for (int j = rowStart + lane; j < rowEnd; j += WARP_SIZE) {
            int col = idx[j];
            sum += val[j] * vec[col];
        }
        vals[t] = sum;
        __syncthreads();

        // Reduce partial sums
        if (lane < 16) vals[t] += vals[t + 16];
        if (lane < 8) vals[t] += vals[t + 8];
        if (lane < 4) vals[t] += vals[t + 4];
        if (lane < 2) vals[t] += vals[t + 2];
        if (lane < 1) vals[t] += vals[t + 1];
        __syncthreads();

        // Write result
        if (lane == 0) {
            res[row] = vals[t];
        }
    }
}

extern "C" __global__ void spmv3(int *cudaRowCounter, int *d_ptr, int *d_cols, int *d_val, float *d_vector, float *d_out, int N) {
    int i;
    float sum;
    int row;
    int rowStart, rowEnd;
    int laneId = threadIdx.x % THREADS_PER_VECTOR;       //lane index in the vector
    int vectorId = threadIdx.x / THREADS_PER_VECTOR;     //vector index in the thread block
    int warpLaneId = threadIdx.x & 31;                   //lane index in the warp
    int warpVectorId = warpLaneId / THREADS_PER_VECTOR;  //vector index in the warp

    __shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

    // Get the row index
    if (warpLaneId == 0) {
        row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
    }
    // Broadcast the value to other threads in the same warp and compute the row index of each vector
    row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

    while (row < N) {
        // Use two threads to fetch the row offset
        if (laneId < 2) {
            space[vectorId][laneId] = d_ptr[row + laneId];
        }
        rowStart = space[vectorId][0];
        rowEnd = space[vectorId][1];

        sum = 0;
        // Compute dot product
        if (THREADS_PER_VECTOR == 32) {
            // Ensure aligned memory access
            i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

            // Process the unaligned part
            if (i >= rowStart && i < rowEnd) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }

            // Process the aligned part
            for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        } else {
            for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        }
        // Intra-vector reduction
        for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }

        // Save the results
        if (laneId == 0) {
            d_out[row] = sum;
        }

        // Get a new row index
        if (warpLaneId == 0) {
            row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
        }
        // Broadcast the row index to the other threads in the same warp and compute the row index of each vector
        row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void sum(const float *x, float *z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i];
    }
    sum = warp_reduce(sum);                    // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum);                     // The first thread in the warp updates the output;
}

extern "C" __global__ void divide(const float *x, float *y, float *val, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] / val[0];
    }
}

extern "C" __global__ void reset_kernel(float *n1, float *n2, int *r1, int *r2) {
    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        *n1 = 0;
        *n2 = 0;
        *r1 = 0;
        *r2 = 0;
    }
}

//////////////////////////////
//////////////////////////////

void CUDART_CB host_callback(void *data) {
    // Check status of GPU after stream operations are done
    callBackData_t *tmp = (callBackData_t *)(data);
    tmp->n1[0] = 0.0;
    tmp->n2[0] = 0.0;
    tmp->r1[0] = 0;
    tmp->r2[0] = 0;
}

//////////////////////////////
//////////////////////////////

void Benchmark7::alloc() {
    nnz = degree * N;
    ptr_tmp = (int *)malloc(sizeof(int) * (N + 1));
    ptr2_tmp = (int *)malloc(sizeof(int) * (N + 1));
    idx_tmp = (int *)malloc(sizeof(int) * nnz);
    idx2_tmp = (int *)malloc(sizeof(int) * nnz);
    val_tmp = (int *)malloc(sizeof(int) * nnz);
    val2_tmp = (int *)malloc(sizeof(int) * nnz);

    err = cudaMallocManaged(&ptr, sizeof(int) * (N + 1));
    err = cudaMallocManaged(&ptr2, sizeof(int) * (N + 1));
    err = cudaMallocManaged(&idx, sizeof(int) * nnz);
    err = cudaMallocManaged(&idx2, sizeof(int) * nnz);
    err = cudaMallocManaged(&val, sizeof(int) * nnz);
    err = cudaMallocManaged(&val2, sizeof(int) * nnz);
    err = cudaMallocManaged(&rowCounter1, sizeof(int));
    err = cudaMallocManaged(&rowCounter2, sizeof(int));

    err = cudaMallocManaged(&auth1, sizeof(float) * N);
    err = cudaMallocManaged(&auth2, sizeof(float) * N);
    err = cudaMallocManaged(&hub1, sizeof(float) * N);
    err = cudaMallocManaged(&hub2, sizeof(float) * N);
    err = cudaMallocManaged(&auth_norm, sizeof(float));
    err = cudaMallocManaged(&hub_norm, sizeof(float));

    x = (int *)malloc(nnz * sizeof(int));
    y = (int *)malloc(nnz * sizeof(int));
    v = (int *)malloc(nnz * sizeof(int));

    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);
}

void Benchmark7::init() {
    random_coo(x, y, v, N, degree);
    // Create a CSR;
    coo2csr(ptr_tmp, idx_tmp, val_tmp, x, y, v, N, N, nnz);
    coo2csr(ptr2_tmp, idx2_tmp, val2_tmp, y, x, v, N, N, nnz);
}

void Benchmark7::reset() {
    // FIXME: using the same data for CSC and CSR, because ptr2 is giving data-dependent performance differences
    for (int j = 0; j < nnz; j++) {
        idx[j] = idx_tmp[j];
        idx2[j] = idx_tmp[j];
        val[j] = val_tmp[j];
        val2[j] = val_tmp[j];
    }
    for (int j = 0; j < N + 1; j++) {
        ptr[j] = ptr_tmp[j];
        ptr2[j] = ptr_tmp[j];
    }
    for (int i = 0; i < N; i++) {
        auth1[i] = 1;
        auth2[i] = 1;
        hub1[i] = 1;
        hub2[i] = 1;
    }
    auth_norm[0] = 0;
    hub_norm[0] = 0;
    rowCounter1[0] = 0;
    rowCounter2[0] = 0;
}

void Benchmark7::execute_sync(int iter) {
    for (int iter = 0; iter < iterations; iter++) {
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(auth1, N * sizeof(float), 0);
            cudaMemPrefetchAsync(auth2, N * sizeof(float), 0);
            cudaMemPrefetchAsync(hub1, N * sizeof(float), 0);
            cudaMemPrefetchAsync(hub2, N * sizeof(float), 0);
            cudaMemPrefetchAsync(auth_norm, sizeof(float), 0);
            cudaMemPrefetchAsync(hub_norm, sizeof(float), 0);
        }

        int nb = ceil(N / ((float)block_size_1d));

        // spmv<<<nb, block_size_1d>>>(ptr2, idx2, val2, hub1, auth2, N, nnz);
        spmv3<<<nb, block_size_1d, block_size_1d * sizeof(float)>>>(rowCounter1, ptr2, idx2, val2, hub1, auth2, N);
        err = cudaDeviceSynchronize();

        // spmv<<<nb, block_size_1d>>>(ptr, idx, val, auth1, hub2, N, nnz);
        spmv3<<<nb, block_size_1d, block_size_1d * sizeof(float)>>>(rowCounter2, ptr, idx, val, auth1, hub2, N);
        err = cudaDeviceSynchronize();

        sum<<<num_blocks, block_size_1d>>>(auth2, auth_norm, N);
        err = cudaDeviceSynchronize();

        sum<<<num_blocks, block_size_1d>>>(hub2, hub_norm, N);
        err = cudaDeviceSynchronize();

        divide<<<num_blocks, block_size_1d>>>(auth2, auth1, auth_norm, N);
        err = cudaDeviceSynchronize();

        divide<<<num_blocks, block_size_1d>>>(hub2, hub1, hub_norm, N);
        err = cudaDeviceSynchronize();

        auth_norm[0] = 0;
        hub_norm[0] = 0;
        rowCounter1[0] = 0;
        rowCounter2[0] = 0;

        if (debug && err) std::cout << err << std::endl;
    }
}

void Benchmark7::execute_async(int iter) {
    if (!pascalGpu || stream_attach) {
        cudaStreamAttachMemAsync(s1, ptr2, 0);
        cudaStreamAttachMemAsync(s1, idx2, 0);
        cudaStreamAttachMemAsync(s1, val2, 0);
        cudaStreamAttachMemAsync(s2, ptr, 0);
        cudaStreamAttachMemAsync(s2, idx, 0);
        cudaStreamAttachMemAsync(s2, val, 0);
    }
    for (int iter = 0; iter < iterations; iter++) {
        if (!pascalGpu || stream_attach) {
            cudaStreamAttachMemAsync(s1, hub1, 0);
            cudaStreamAttachMemAsync(s1, auth2, 0);
            cudaStreamAttachMemAsync(s2, auth1, 0);
            cudaStreamAttachMemAsync(s2, hub2, 0);
        }
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(auth1, N * sizeof(float), 0, s2);
            cudaMemPrefetchAsync(auth2, N * sizeof(float), 0, s1);
            cudaMemPrefetchAsync(hub1, N * sizeof(float), 0, s1);
            cudaMemPrefetchAsync(hub2, N * sizeof(float), 0, s2);
            cudaMemPrefetchAsync(auth_norm, sizeof(float), 0, s1);
            cudaMemPrefetchAsync(hub_norm, sizeof(float), 0, s2);
        }

        cudaEvent_t e1, e2;
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);

        int nb = ceil(N / ((float)block_size_1d));

        // spmv<<<nb, block_size_1d, 0, s1>>>(ptr2, idx2, val2, hub1, auth2, N, nnz);
        spmv3<<<nb, block_size_1d, block_size_1d * sizeof(float), s1>>>(rowCounter1, ptr2, idx2, val2, hub1, auth2, N);
        err = cudaEventRecord(e1, s1);
        // spmv<<<nb, block_size_1d, 0, s2>>>(ptr, idx, val, auth1, hub2, N, nnz);
        spmv3<<<nb, block_size_1d, block_size_1d * sizeof(float), s2>>>(rowCounter2, ptr, idx, val, auth1, hub2, N);
        err = cudaEventRecord(e2, s2);

        sum<<<num_blocks, block_size_1d, 0, s1>>>(auth2, auth_norm, N);

        sum<<<num_blocks, block_size_1d, 0, s2>>>(hub2, hub_norm, N);

        // Stream 1 waits stream 2;
        err = cudaStreamWaitEvent(s1, e2, 0);
        cudaStreamAttachMemAsync(s1, auth1, 0);
        divide<<<num_blocks, block_size_1d, 0, s1>>>(auth2, auth1, auth_norm, N);
        // Stream 2 waits stream 1;
        err = cudaStreamWaitEvent(s2, e1, 0);
        cudaStreamAttachMemAsync(s2, hub1, 0);
        divide<<<num_blocks, block_size_1d, 0, s2>>>(hub2, hub1, hub_norm, N);

        // cudaEvent_t e3;
        // cudaEventCreate(&e3);
        // cudaEventRecord(e3, s2);
        // checkCudaErrors(cudaStreamWaitEvent(s1, e3, 0));
        // reset_kernel<<<1, 1, 0, s1>>>(auth_norm, hub_norm, rowCounter1, rowCounter2);

        err = cudaStreamSynchronize(s1);
        err = cudaStreamSynchronize(s2);
        auth_norm[0] = 0;
        hub_norm[0] = 0;
        rowCounter1[0] = 0;
        rowCounter2[0] = 0;

        if (debug && err) std::cout << err << std::endl;
    }
    // err = cudaStreamSynchronize(s1);
}

void Benchmark7::execute_cudagraph(int iter) {
    if (iter == 0) {
        cudaEvent_t ef;
        cudaEventCreate(&ef);
        cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
        cudaEventRecord(ef, s1);
        cudaStreamWaitEvent(s2, ef, 0);

        // callBackData_t hostFnData = {auth_norm, hub_norm, rowCounter1, rowCounter2};
        // cudaHostFn_t fn = host_callback;

        for (int i = 0; i < iterations; i++) {
            cudaEvent_t e1, e2;
            cudaEventCreate(&e1);
            cudaEventCreate(&e2);

            int nb = ceil(N / ((float)block_size_1d));

            // spmv<<<nb, block_size_1d, 0, s1>>>(ptr2, idx2, val2, hub1, auth2, N, nnz);
            spmv3<<<nb, block_size_1d, block_size_1d * sizeof(float), s1>>>(rowCounter1, ptr2, idx2, val2, hub1, auth2, N);

            // spmv<<<nb, block_size_1d, 0, s2>>>(ptr, idx, val, auth1, hub2, N, nnz);
            spmv3<<<nb, block_size_1d, block_size_1d * sizeof(float), s2>>>(rowCounter2, ptr, idx, val, auth1, hub2, N);

            sum<<<num_blocks, block_size_1d, 0, s1>>>(auth2, auth_norm, N);
            err = cudaEventRecord(e1, s1);
            sum<<<num_blocks, block_size_1d, 0, s2>>>(hub2, hub_norm, N);
            err = cudaEventRecord(e2, s2);
            // Stream 1 waits stream 2;
            err = cudaStreamWaitEvent(s1, e2, 0);
            divide<<<num_blocks, block_size_1d, 0, s1>>>(auth2, auth1, auth_norm, N);
            // Stream 2 waits stream 1;
            err = cudaStreamWaitEvent(s2, e1, 0);
            divide<<<num_blocks, block_size_1d, 0, s2>>>(hub2, hub1, hub_norm, N);
            // Stream 1 waits stream 2;
            cudaEvent_t e3;
            cudaEventCreate(&e3);
            cudaEventRecord(e3, s2);
            checkCudaErrors(cudaStreamWaitEvent(s1, e3, 0));

            // This doesn't work for some reason;
            // checkCudaErrors(cudaLaunchHostFunc(s1, fn, &hostFnData));

            reset_kernel<<<1, 1, 0, s1>>>(auth_norm, hub_norm, rowCounter1, rowCounter2);
            cudaEvent_t e4;
            cudaEventCreate(&e4);
            cudaEventRecord(e4, s1);
            checkCudaErrors(cudaStreamWaitEvent(s2, e4, 0));
        }

        checkCudaErrors(cudaStreamEndCapture(s1, &graph));
        checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    }
    checkCudaErrors(cudaGraphLaunch(graphExec, s1));
    err = cudaStreamSynchronize(s1);
}

void Benchmark7::execute_cudagraph_manual(int iter) {
    if (iter == 0) {
        // callBackData_t hostFnData = {auth_norm, hub_norm, rowCounter1, rowCounter2};
        // cudaHostFn_t fn = host_callback;

        int pascalGpu = 0;
        // cudaDeviceGetAttribute(&pascalGpu, cudaDeviceAttr::cudaDevAttrConcurrentManagedAccess, 0);
        cudaGraphCreate(&graph, 0);
        void *kernel_1_args[7] = {(void *)&rowCounter1, (void *)&ptr2, (void *)&idx2, (void *)&val2, (void *)&hub1, (void *)&auth2, &N};
        void *kernel_2_args[7] = {(void *)&rowCounter2, (void *)&ptr, (void *)&idx, (void *)&val, (void *)&auth1, (void *)&hub2, &N};
        void *kernel_3_args[3] = {(void *)&auth2, (void *)&auth_norm, &N};
        void *kernel_4_args[3] = {(void *)&hub2, (void *)&hub_norm, &N};
        void *kernel_5_args[4] = {(void *)&auth2, (void *)&auth1, (void *)&auth_norm, &N};
        void *kernel_6_args[4] = {(void *)&hub2, (void *)&hub1, (void *)&hub_norm, &N};
        void *kernel_7_args[4] = {(void *)&auth_norm, (void *)&hub_norm, (void *)&rowCounter1, (void *)&rowCounter2};

        callback_data = {0};
        callback_data.n1 = auth_norm;
        callback_data.n2 = hub_norm;
        callback_data.r1 = rowCounter1;
        callback_data.r2 = rowCounter2;

        for (int i = 0; i < iterations; i++) {
            dim3 tb(block_size_1d);
            dim3 bs(num_blocks);
            dim3 nb(ceil(N / ((float)block_size_1d)));

            if (i > 0) {
                nodeDependencies.clear();
                if (pascalGpu) {
                    nodeDependencies.push_back(host_node);
                } else {
                    nodeDependencies.push_back(kernel_7);
                }
            }
            checkCudaErrors(add_node(kernel_1_args, kernel_1_params, (void *)spmv3, nb, tb, graph, &kernel_1, nodeDependencies, block_size_1d * sizeof(float)));
            if (i > 0) {
                nodeDependencies.clear();
                if (pascalGpu) {
                    nodeDependencies.push_back(host_node);
                } else {
                    nodeDependencies.push_back(kernel_7);
                }
            }
            add_node(kernel_2_args, kernel_2_params, (void *)spmv3, nb, tb, graph, &kernel_2, nodeDependencies, block_size_1d * sizeof(float));

            nodeDependencies.clear();
            nodeDependencies.push_back(kernel_1);
            add_node(kernel_3_args, kernel_3_params, (void *)sum, bs, tb, graph, &kernel_3, nodeDependencies);

            nodeDependencies.clear();
            nodeDependencies.push_back(kernel_2);
            add_node(kernel_4_args, kernel_4_params, (void *)sum, bs, tb, graph, &kernel_4, nodeDependencies);

            nodeDependencies.clear();
            nodeDependencies.push_back(kernel_2);
            nodeDependencies.push_back(kernel_3);
            add_node(kernel_5_args, kernel_5_params, (void *)divide, bs, tb, graph, &kernel_5, nodeDependencies);

            nodeDependencies.clear();
            nodeDependencies.push_back(kernel_1);
            nodeDependencies.push_back(kernel_4);
            checkCudaErrors(add_node(kernel_6_args, kernel_6_params, (void *)divide, bs, tb, graph, &kernel_6, nodeDependencies));

            nodeDependencies.clear();
            nodeDependencies.push_back(kernel_5);
            nodeDependencies.push_back(kernel_6);
            if (pascalGpu) {
                host_params.fn = host_callback;
                host_params.userData = (void *)&callback_data;

                checkCudaErrors(cudaGraphAddHostNode(&host_node, graph,
                                                     nodeDependencies.data(),
                                                     nodeDependencies.size(), &host_params));
            } else {
                add_node(kernel_7_args, kernel_7_params, (void *)reset_kernel, bs, tb, graph, &kernel_7, nodeDependencies);
            }

            // spmv3<<<nb, block_size_1d, block_size_1d * sizeof(float), s1>>>(rowCounter1, ptr2, idx2, val2, hub1, auth2, N);
            // spmv3<<<nb, block_size_1d, block_size_1d * sizeof(float), s2>>>(rowCounter2, ptr, idx, val, auth1, hub2, N);
            // sum<<<num_blocks, block_size_1d, 0, s1>>>(auth2, auth_norm, N);
            // sum<<<num_blocks, block_size_1d, 0, s2>>>(hub2, hub_norm, N);
            // divide<<<num_blocks, block_size_1d, 0, s1>>>(auth2, auth1, auth_norm, N);
            // divide<<<num_blocks, block_size_1d, 0, s2>>>(hub2, hub1, hub_norm, N);

            // reset_kernel<<<1, 1, 0, s1>>>(auth_norm, hub_norm, rowCounter1, rowCounter2);
        }
        checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    }
    checkCudaErrors(cudaGraphLaunch(graphExec, s1));
    err = cudaStreamSynchronize(s1);
}

void Benchmark7::execute_cudagraph_single(int iter) {
    if (iter == 0) {
        cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);

        for (int i = 0; i < iterations; i++) {
            int nb = ceil(N / ((float)block_size_1d));

            // spmv<<<nb, block_size_1d, 0, s1>>>(ptr2, idx2, val2, hub1, auth2, N, nnz);
            spmv3<<<nb, block_size_1d, block_size_1d * sizeof(float), s1>>>(rowCounter1, ptr2, idx2, val2, hub1, auth2, N);

            // spmv<<<nb, block_size_1d, 0, s2>>>(ptr, idx, val, auth1, hub2, N, nnz);
            spmv3<<<nb, block_size_1d, block_size_1d * sizeof(float), s1>>>(rowCounter2, ptr, idx, val, auth1, hub2, N);

            sum<<<num_blocks, block_size_1d, 0, s1>>>(auth2, auth_norm, N);
            sum<<<num_blocks, block_size_1d, 0, s1>>>(hub2, hub_norm, N);

            divide<<<num_blocks, block_size_1d, 0, s1>>>(auth2, auth1, auth_norm, N);

            divide<<<num_blocks, block_size_1d, 0, s1>>>(hub2, hub1, hub_norm, N);

            reset_kernel<<<1, 1, 0, s1>>>(auth_norm, hub_norm, rowCounter1, rowCounter2);
        }

        checkCudaErrors(cudaStreamEndCapture(s1, &graph));
        checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    }
    checkCudaErrors(cudaGraphLaunch(graphExec, s1));
    err = cudaStreamSynchronize(s1);
}

std::string Benchmark7::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(auth1[0]);
    } else {
        std::string res = "[";
        for (int j = 0; j < 10; j++) {
            res += std::to_string(auth1[j]) + ", ";
        }
        return res + ", ...]";
    }
}