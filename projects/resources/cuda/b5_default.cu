#include <chrono>
#include <cstdlib>
#include <ctime>
#include <math.h> 
#include <iostream>
#include "utils.hpp"
#include "options.hpp"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

// float R = 0.08;
// float V = 0.3;
// float T = 1.0;
// float K = 60.0;

// __device__ inline float cndGPU(float d) {
//     const float       A1 = 0.31938153f;
//     const float       A2 = -0.356563782f;
//     const float       A3 = 1.781477937f;
//     const float       A4 = -1.821255978f;
//     const float       A5 = 1.330274429f;
//     const float RSQRT2PI = 0.39894228040143267793994605993438f;

//     float
//     K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

//     float
//     cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
//           (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

//     if (d > 0)
//         cnd = 1.0f - cnd;

//     return cnd;
// }

// extern "C" __global__ void bs(const float *x, float *y, int N, float R, float V, float T, float K) {

//     float sqrtT = __fdividef(1.0F, rsqrtf(T));
//     for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
//         float expRT;
//         float d1, d2, CNDD1, CNDD2;
//         d1 = __fdividef(__logf(x[i] / K) + (R + 0.5f * V * V) * T, V * sqrtT);
//         d2 = d1 - V * sqrtT;

//         CNDD1 = cndGPU(d1);
//         CNDD2 = cndGPU(d2);

//         //Calculate Call and Put simultaneously
//         expRT = __expf(-R * T);
//         y[i] = x[i] * CNDD1 - K * expRT * CNDD2;
//     }
// }

double R = 0.08;
double V = 0.3;
double T = 1.0;
double K = 60.0;

__device__ inline double cndGPU(double d) {
    const double       A1 = 0.31938153f;
    const double       A2 = -0.356563782f;
    const double       A3 = 1.781477937f;
    const double       A4 = -1.821255978f;
    const double       A5 = 1.330274429f;
    const double RSQRT2PI = 0.39894228040143267793994605993438f;

    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

extern "C" __global__ void bs(const double *x, double *y, int N, double R, double V, double T, double K) {

    double sqrtT = 1.0 / rsqrt(T);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        double expRT;
        double d1, d2, CNDD1, CNDD2;
        d1 = (log(x[i] / K) + (R + 0.5 * V * V) * T) / (V * sqrtT);
        d2 = d1 - V * sqrtT;

        CNDD1 = cndGPU(d1);
        CNDD2 = cndGPU(d2);

        //Calculate Call and Put simultaneously
        expRT = exp(-R * T);
        y[i] = x[i] * CNDD1 - K * expRT * CNDD2;
    }
}

/////////////////////////////
/////////////////////////////

void init(double **x, double **y, double* tmp_x, int N, int K) {
    for (int j = 0; j < N; j++) {
        tmp_x[j] = 60 - 0.5 + (double) rand() / RAND_MAX;
        for (int i = 0; i < K; i++) {
            x[i][j] = tmp_x[j];
            // y[i][j] = 0;
        }
    }
}

void reset(double **x, double* y, int N, int K, cudaStream_t *s) {
    for (int i = 0; i < K; i++) {
        // memcpy(x[i], y, sizeof(int) * N);
        // cudaMemcpy(x[i], y, sizeof(double) * N, cudaMemcpyDefault);
        
        // cudaMemcpyAsync(x[i], y, sizeof(int) * N, cudaMemcpyHostToDevice, s[i]);
        for (int j = 0; j < N; j++) {
            x[i][j] = y[j];
        }
    }
    // cudaMemPrefetchAsync(x[0], sizeof(double) * N, 0, s[0]);
}


/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

    srand(time(0));

    Options options = Options(argc, argv);
	int debug = options.debug;

    int num_executions = options.num_iter;
    int N = options.N;

    int block_size = options.block_size_1d;
    int num_blocks = options.num_blocks;
    int skip_iterations = options.skip_iterations;
    int err = 0;

    int M = 10;

    if (debug) {
        std::cout << "running b5 default" << std::endl;
        std::cout << "N=" << N << std::endl;
        std::cout << "num executions=" << num_executions << std::endl;
        std::cout << "block size 1d=" << block_size << std::endl;
        std::cout << "num blocks=" << num_blocks << std::endl;
        std::cout << "skip iteration time=" << skip_iterations << std::endl;
    }
    
    auto start = clock_type::now();
    double **x = (double **) malloc(sizeof(double*) * M);
    double **y = (double **) malloc(sizeof(double*) * M);
    double *tmp_x = (double *) malloc(sizeof(double) * N);
    // cudaHostRegister(tmp_x, sizeof(double) * N, 0);

    for (int i = 0; i < M; i++) {
        cudaMallocManaged(&x[i], sizeof(double) * N);
        cudaMallocManaged(&y[i], sizeof(double) * N);
    }
    if (debug && err) std::cout << err << std::endl;
    
    // Create streams;
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * M);
    for (int i = 0; i < M; i++) {
        err = cudaStreamCreate(&s[i]);
    }
    if (debug && err) std::cout << err << std::endl;

    // Initialze arrays;
    init(x, y, tmp_x, N, M);

    if (debug) std::cout << "x[0][0]=" << tmp_x[0] << std::endl;

    auto end = clock_type::now();
    if (debug) std::cout << "init=" << (double) chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    // Print header;
    if (!debug) std::cout << "num_iter,gpu_result,total_time_sec,overhead_sec,computation_sec" << std::endl;
	
    double tot = 0;
    for (int i = 0; i < num_executions; i++) {
        if (debug) std::cout << "\n-- iter=" << i << std::endl;
        auto start_tmp = clock_type::now();
        reset(x, tmp_x, N, M, s);
        auto end_tmp = clock_type::now();
        auto reset_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        if (debug) std::cout << "  reset=" << (double) reset_time / 1000 << " ms" << std::endl;
        
        start = clock_type::now();

        for (int j = 0; j < M; j++) {
            cudaStreamAttachMemAsync(s[j], x[j], sizeof(double) * N);
            cudaStreamAttachMemAsync(s[j], y[j], sizeof(double) * N);
            // if (j > 0) cudaMemPrefetchAsync(y[j - 1], sizeof(double) * N, cudaCpuDeviceId, s[j - 1]);
            bs<<<num_blocks, block_size, 0, s[j]>>>(x[j], y[j], N, R, V, T, K);
            // if (j < M - 1) cudaMemPrefetchAsync(x[j + 1], sizeof(double) * N, 0, s[j + 1]);
        }

        // Last tile;
        // cudaMemPrefetchAsync(y[M - 1], sizeof(double) * N, cudaCpuDeviceId, s[M - 1]);

        for (int j = 0; j < M; j++) {
            err = cudaStreamSynchronize(s[j]);
        }

        if (debug && err) std::cout << err << std::endl;

        end = clock_type::now();
        auto tmp = chrono::duration_cast<chrono::microseconds>(end - start).count();
        if (i >= skip_iterations) tot += tmp;

        if (debug) {
            std::cout << "  gpu result=[";
            for (int j = 0; j < M; j++) {
                std::cout << y[j][0] << ", ";
            } 
            std::cout << ", ...]; time=" << (double) tmp / 1000 << " ms" << std::endl;
        } else {
            std::cout << i << "," << y[0][0] << "," << (double) (reset_time + tmp) / 1e6 << "," << (double) reset_time / 1e6 << "," << (double) tmp / 1e6 << std::endl;
        }
    }

    // Print;
	cudaDeviceSynchronize();
    
    if (debug) std::cout << "\nmean exec time=" << (double) tot / (1000 * (num_executions - skip_iterations)) << " ms" << std::endl;
}
