#include <iostream>
#include <stdio.h>
#include <nvml.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <fstream>

#define N 500000000 // 500 MB

// Test bandwidth between two GPUs;
float dtod_copy(size_t size, int from, int to) {
	int *pointers[2];

	cudaSetDevice(from);
	cudaDeviceEnablePeerAccess(to, 0);
	cudaMalloc(&pointers[0], size);

	cudaSetDevice(to);
	cudaDeviceEnablePeerAccess(from, 0);
	cudaMalloc(&pointers[1], size);

	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	cudaEventRecord(begin);
	cudaMemcpyAsync(pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float elapsed;
	cudaEventElapsedTime(&elapsed, begin, end);
	elapsed /= 1000;

	cudaSetDevice(from);
	cudaFree(pointers[0]);

	cudaSetDevice(to);
	cudaFree(pointers[1]);

	cudaEventDestroy(end);
	cudaEventDestroy(begin);
	cudaSetDevice(from);

	return elapsed;
}

// Test bandwidth from the CPU to a device;
float htod_copy(size_t size, int device_id) {
	int *pointer, *d_pointer;

	cudaSetDevice(device_id);
	cudaMalloc(&d_pointer, size);
	cudaMallocHost(&pointer, size);

	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	cudaEventRecord(begin);
	cudaMemcpyAsync(d_pointer, pointer, size, cudaMemcpyHostToDevice);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float elapsed;
	cudaEventElapsedTime(&elapsed, begin, end);
	elapsed /= 1000;

	cudaSetDevice(device_id);
	cudaFree(d_pointer);

	cudaEventDestroy(end);
	cudaEventDestroy(begin);

	return elapsed;
}

int main() {
    int gpu_number = 0;

	cudaGetDeviceCount(&gpu_number);  
	printf("number of devices = %d\n", gpu_number);

	double **bandwidths = (double**) malloc(gpu_number * sizeof(double*));
	for (int i = 0; i < gpu_number; i++) {
		bandwidths[i] = (double*) malloc(gpu_number * sizeof(double));
    }
	std::ofstream out_file;
	// This is not safe, I guess;
    std::string grcuda_home = getenv("GRCUDA_HOME");
	out_file.open(grcuda_home + "/projects/resources/connection_graph/datasets/connection_graph.csv");
	out_file << "From,To,Bandwidth\n";

	for (int i = 0; i < gpu_number; i++) {
        // Measure CPU-to-GPU transfer time;
		double time_htod = htod_copy(N, 1);
		printf("\nfrom: Host, to: %d, time spent: %f, transfer rate: %f GB/s \n",i, time_htod, (float(N) / 1000000000.0) / time_htod);
		out_file << std::setprecision(15) << "-1" << "," << i << "," << (double(N) /1000000000.0) / time_htod << "\n";
		
        for (int j = 0 ; j < gpu_number; j++) {
            // Measure GPU-to-GPU transfer time;
			double time_dtod = dtod_copy(N, i, j);
			bandwidths[i][j] = (double(N) / 1000000000.0) / time_dtod;
			printf("from: %d, to: %d, time spent: %f, transfer rate: %f GB/s \n", i, j, time_dtod, bandwidths[i][j]);
			out_file << i << "," << j << "," << bandwidths[i][j] << "\n";
		}
	}
	out_file.close();
	return 0;
}
