#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:25:57 2021
@author: alberto.parravicini
"""

import polyglot
from java.lang import System 
import numpy as np
from random import random, randint, seed, sample, uniform

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D, DEFAULT_BLOCK_SIZE_2D, DEFAULT_NUM_BLOCKS
from benchmark_result import BenchmarkResult

##############################
##############################

MATRIX_VECTOR_MULT_KERNEL = """   
extern "C" __global__ void matrix_vector_mult_1(const float* x, const float* y, float* z, int n, int m, int z_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[z_offset + i] = sum;
    }
}

extern "C" __global__ void matrix_vector_mult_2(const float* x, const float* y, float* z, int n, int m) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[i] = sum;
    }
}

extern "C" __global__ void copy(const float *x, float *y, int n, int offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i + offset] = x[i];
    }
}
"""

'''THIS IS EMPLOYED AS DEFAULT BENCHMARK FOR MULTIGPU TEST'''

class Benchmark11M(Benchmark):
    """
    Dense matrix-vector multiplication, partitioning the matrix in blocks of rows;
    """

    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b11m", benchmark, nvprof_profile)
        self.size = 0
        
        # Square matrix of size x size;
        self.N = self.size
        self.M = self.size
        
        # Use P horizontal partitions;
        self.P = 16
        
        # Size of partitions;
        self.S = (self.N + self.P - 1) // self.P
        
        # Full matrix;
        self.x_cpu = None
        # Dense vector;
        self.y_cpu = None
        
        # The GPU matrix is stored using P arrays;
        self.x = [None for _ in range(self.P)]
        # Dense vector;
        self.y = None
        # Result;
        # self.z = None        
        self.z = [None for _ in range(self.P)]
        self.z_out = None

        self.cpu_result = None
        self.gpu_result = None

        self.block_size_1d = DEFAULT_BLOCK_SIZE_1D
        self.block_size_2d = DEFAULT_BLOCK_SIZE_2D

        self.matrix_vector_mult_kernel = None

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.N = self.size
        self.M = self.size
        self.S = (self.N + self.P - 1) // self.P
        self.block_size_1d = block_size["block_size_1d"]
        self.block_size_2d = block_size["block_size_2d"]

        self.gpu_result = 0.0

        # Allocate vectors;
        for p in range(self.P):
            self.x[p] = polyglot.eval(language="grcuda", string=f"float[{self.S * self.M}]")
        self.y = polyglot.eval(language="grcuda", string=f"float[{self.M}]")
        # self.z = polyglot.eval(language="grcuda", string=f"float[{self.N}]")
        for p in range(self.P):
            self.z[p] = polyglot.eval(language="grcuda", string=f"float[{self.S}]")
        self.z_out = polyglot.eval(language="grcuda", string=f"float[{self.N}]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        # self.matrix_vector_mult_kernel = build_kernel(MATRIX_VECTOR_MULT_KERNEL, "matrix_vector_mult_2", "const pointer, const pointer, pointer, sint32, sint32, sint32")
        self.matrix_vector_mult_kernel = build_kernel(MATRIX_VECTOR_MULT_KERNEL, "matrix_vector_mult_2", "const pointer, const pointer, pointer, sint32, sint32")
        self.copy_kernel = build_kernel(MATRIX_VECTOR_MULT_KERNEL, "copy", "const pointer, pointer, sint32, sint32")
        self.initialize = polyglot.eval(language="js", string="x => { for (let i = 0; i < x.length; i++) { x[i] = i / x.length }}")

    @time_phase("initialization")
    def init(self):
        self.random_seed = 10
        seed(self.random_seed)

    @time_phase("reset_result")
    def reset_result(self) -> None:
        self.gpu_result = 0.0
        for p in range(self.P):
            self.initialize(self.x[p])
        for i in range(self.M):
            self.y[i] = i / self.M
        # for p in range(self.P):
        #     for i in range(len(self.x[p])):
        #         print(f"p={p}, x[{p}][{i}]={self.x[p][i]}")
        # for i in range(len(self.y)):
        #     print(f"i={i}, y[{i}]={self.y[i]}")
        
    def execute(self) -> object:
        self.num_blocks = self.num_blocks 
        self.block_size = self._block_size["block_size_1d"]
        # Schedule the categorical Naive Bayes and Ridge Regression kernels
        start_comp = System.nanoTime()
        start = 0

        # Compute all partitions;
        for p in range(self.P):
            # self.execute_phase("mmul_{p}", self.matrix_vector_mult_kernel(self.num_blocks, self.block_size),
            #                    self.x[p], self.y, self.z, min(self.S, self.N - p * self.S), self.M, p * self.S)
            self.execute_phase("mmul_{p}", self.matrix_vector_mult_kernel(self.num_blocks, self.block_size),
                               self.x[p], self.y, self.z[p], min(self.S, self.N - p * self.S), self.M)
        # Aggregate results;
        for p in range(self.P):      
            self.execute_phase("copy_{p}", self.copy_kernel(self.num_blocks, self.block_size),
                               self.z[p], self.z_out, min(self.S, self.N - p * self.S), p * self.S)      

        # Add a final sync step to measure the real computation time;
        if self.time_phases:
            start = System.nanoTime()
        tmp = self.z_out[0]
        end = System.nanoTime()
        self.gpu_result = sum(self.z_out[:10])
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)
        self.benchmark.add_to_benchmark("gpu_result", self.gpu_result)
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: [" + ", ".join([f"{x:.4f}" for x in self.z_out[:10]]) + "...]")

        return self.gpu_result   

    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:
        start = System.nanoTime()
        x_cpu = [0.0] * self.N * self.M
        y_cpu = [self.y[i] for i in range(len(self.y))]
        for i in range(self.P):
            for j in range(self.S * self.M):
                if i * self.S * self.M + j < len(x_cpu):
                    x_cpu[i * self.S * self.M + j] = j / (self.S * self.M)
        z_cpu = np.array(x_cpu).reshape((self.N, self.M)) @ np.array(y_cpu)
        self.cpu_result = sum(z_cpu[:10])
        cpu_time = System.nanoTime() - start
            
        # Compare GPU and CPU results;
        difference = np.abs(self.cpu_result - gpu_result)

        self.benchmark.add_to_benchmark("cpu_time_sec", cpu_time)
        self.benchmark.add_to_benchmark("cpu_gpu_res_difference", str(difference))
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tcpu result: [" + ", ".join([f"{x:.4f}" for x in z_cpu[:10]]) + "...]; " +
                                        f"difference: {difference:.4f}, time: {cpu_time:.4f} sec")






