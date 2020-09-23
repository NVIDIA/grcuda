extern "C" __global__ void nb_1(const int* x, const float* y, float* z, int size, int n_feat, int n_classes) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_classes; j++) {
            for (int q = 0; q < n_feat; q++) {
                z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
            } 
        }
    }
}

extern "C" __global__ void nb_2(const float* x, float* y, int n_row_x, int n_col_x) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float curr_max = x[i * n_col_x];
        for (int j = 0; j < n_col_x; j++) {
            curr_max = fmaxf(curr_max, x[i * n_col_x + j]); 
        }
        y[i] = curr_max;
    }
}

extern "C" __global__ void nb_3(const float* x, const float* y, float* z, int n_row_x, int n_col_x) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < n_col_x; j++) {
            sum += expf(x[i * n_col_x + j] - y[i]);
        }
        z[i] = logf(sum) + y[i];
    }
}

extern "C" __global__ void nb_4(float* x, float* y, int n_row_x, int n_col_x) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] = expf(x[i * n_col_x + j] - y[i]);
        }
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void rr_1_0(const int* x, float *y, float *z, int n_row_x, int n_col_x) {
    int warp_size = 32;
    for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {
        // Compute mean and variance;
        float feature_mean = float(0);
        float sum_sq = float(0);
        for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < n_row_x; i += blockDim.y * gridDim.y) {
            float x_tmp = x[j * n_row_x + i];
            feature_mean += x_tmp;
            sum_sq += x_tmp * x_tmp;
        }
        feature_mean = warp_reduce(feature_mean); // Obtain the sum of values in the current warp;
        sum_sq = warp_reduce(sum_sq); // Obtain the sum of values in the current warp;
        if (!(threadIdx.y % warp_size)) {
            atomicAdd(y + j, feature_mean); 
            atomicAdd(z + j, sum_sq); 
        }
    }
}

extern "C" __global__ void rr_1_1(const int *x, float *y, const float *mean, const float *std, int n_row_x, int n_col_x) {
    for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {
        float mean_tmp = mean[j] / n_row_x;
        float std_tmp = sqrtf(std[j] / n_row_x - mean_tmp * mean_tmp);
        
        for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < n_row_x; i += blockDim.y * gridDim.y) {
            y[j * n_row_x + i] = ((float) x[j * n_row_x + i] - mean_tmp) / std_tmp; 
        }
    }
}

extern "C" __global__ void rr_1(const int* x, float *y, int n_row_x, int n_col_x) {
    for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {
        float feature_mean = 0;
        float sum_sq = 0;
        // Compute mean and variance;
        for (int i = 0; i < n_row_x; i++) {
            float x_tmp = x[j * n_row_x + i];
            feature_mean += x_tmp;
            sum_sq += x_tmp * x_tmp;
        }
        feature_mean /= n_row_x;
        float std = sqrtf(sum_sq / n_row_x - feature_mean * feature_mean);
        
        // Update values;
        for (int i = 0; i < n_row_x; i++) {
            y[j * n_row_x + i] = (x[j * n_row_x + i] - feature_mean) / std;
        }
    }
}

extern "C" __global__ void rr_2(const float* x, const float* y, float* z, int size, int n_feat, int n_classes) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_classes; j++) {
            for (int q = 0; q < n_feat; q++) {
                z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
            }
        }
    }
}

extern "C" __global__ void rr_3(float* x, const float *y, int n_row_x, int n_col_x) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] += y[j];
        }
    }
}

extern "C" __global__ void softmax(float *x, int n_row_x, int n_col_x) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        float row_exp_sum = 0;
        for (int j = 0; j < n_col_x; j++) {
            row_exp_sum += expf( x[i * n_col_x + j]);
        }
        for (int j = 0; j < n_col_x; j++) {
             x[i * n_col_x + j] = expf(x[i * n_col_x + j]) / row_exp_sum;
        }
    }
}

extern "C" __global__ void argmax(const float *x, const float *y, int *z, int n_row_x, int n_col_x) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {
        int curr_best_index = 0;
        float curr_best = x[i * n_col_x] + y[i * n_col_x];
        for (int j = 0; j < n_col_x; j++) {
            float curr = x[i * n_col_x + j] + y[i * n_col_x + j];
            if (curr > curr_best) {
                curr_best = curr;
                curr_best_index = j;
            }
        }
        z[i] = curr_best_index;
    }
}
