
#define WARP_SIZE 32
#define THREADS_PER_VECTOR 4
#define MAX_NUM_VECTORS_PER_BLOCK (1024 / THREADS_PER_VECTOR)

/////////////////////////////
/////////////////////////////

extern "C" __global__ void spmv(const int *ptr, const int *idx, const int *val, const float *vec, float *res, int num_rows, int num_nnz) {

    for(int n = blockIdx.x * blockDim.x + threadIdx.x; n < num_rows; n += blockDim.x * gridDim.x) {
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
        int rowEnd = ptr[row+1];
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
        if (lane <  8) vals[t] += vals[t + 8];
        if (lane <  4) vals[t] += vals[t + 4];
        if (lane <  2) vals[t] += vals[t + 2];
        if (lane <  1) vals[t] += vals[t + 1];
        __syncthreads();

        // Write result
        if (lane == 0) {
            res[row] = vals[t];
        }
    }	
}

extern "C" __global__ void spmv3(int* cudaRowCounter, int* d_ptr, int* d_cols, int* d_val, float* d_vector, float* d_out, int N) {
	int i;
	float sum;
	int row;
	int rowStart, rowEnd;
	int laneId = threadIdx.x % THREADS_PER_VECTOR; //lane index in the vector
	int vectorId = threadIdx.x / THREADS_PER_VECTOR; //vector index in the thread block
	int warpLaneId = threadIdx.x & 31;	//lane index in the warp
	int warpVectorId = warpLaneId / THREADS_PER_VECTOR;	//vector index in the warp

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
            sum += __shfl_down_sync(0xffffffff,sum, i);
		}

		// Save the results
		if (laneId == 0) {
			d_out[row] = sum;
		}

		// Get a new row index
		if(warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		// Broadcast the row index to the other threads in the same warp and compute the row index of each vector
        row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;
	}
}


__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void sum(const float *x, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

extern "C" __global__ void divide(const float* x, float *y, float *val, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] / val[0];
    }
}