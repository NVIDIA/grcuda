
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