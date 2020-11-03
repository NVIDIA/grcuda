// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "trt.h"

//
// Simple application that uses libtrt. It serializes an engine
// loads a pgm image and performance a single inference job.
//

#define CUDA_TRY(status) cudaTry_(status, __FILE__, __LINE__)
static void cudaTry_(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s\n", file, line, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


float* read_pgm_file(const char * const file_name, int *width, int *height) {
  FILE *image_file = fopen(file_name, "r");
  if (image_file == NULL) {
    perror("unable to open image file");
    return NULL;
  }
  char magic[3];
  fscanf(image_file, "%s", magic);
  if (strncmp(magic, "P5", 2) != 0) {
    fprintf(stderr, "invalid magic header in image file: %s\n", magic);
    exit(EXIT_FAILURE);
  }
  int max_gray_level;
  fscanf(image_file, "%d %d %d", width, height, &max_gray_level);
  // printf("width: %d\n", *width);
  // printf("height: %d\n", *height);
  // printf("max_level: %d\n", max_gray_level);
  size_t offset = ftell(image_file) + 1;
  fseek(image_file, 0, SEEK_END);
  size_t size_bytes = ftell(image_file) - offset;
  fseek(image_file, offset, SEEK_SET);
  // printf("size: %lu\n", size_bytes);
  const int element_size = (max_gray_level > 255) ? 2 : 1;
  if (size_bytes != element_size * *width * *height) {
    fprintf(stderr, "invalid file size, expected %d bytes got %lu bytes\n",
      element_size * *width * *height, size_bytes);
    exit(EXIT_FAILURE);
  }

  char* buf = malloc(size_bytes);
  if (fread(buf, 1, size_bytes, image_file) != size_bytes) {
    fprintf(stderr, "invalid read\n");
    exit(EXIT_FAILURE);
  }

  float *img = malloc(sizeof(float) * *width * *height);
  for (int y = 0; y < *height; ++y) {
    for (int x = 0; x < *width; ++x) {
      int idx = y * *width + x;
      float value = (element_size == 1) ? (((unsigned char*)buf)[idx]) : (((unsigned short*)buf)[idx]);
      img[idx] = 1.0f - value / max_gray_level;
    }
  }
  free(buf);
  return img;
}

#define NUM_OUTPUT_ELEMENTS 10

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s model.engine image.pgm\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  const char * const engine_file_name = argv[1];
  const char * const image_file_name = argv[2];

  printf("creating inference runtime...\n");
  runtime_handle_t runtime = createInferRuntime();
  if (runtime < 0) {
    fprintf(stderr, "unable to create runtime\n");
    exit(EXIT_FAILURE);
  }
  int return_code = -1;
  trt_error_t res;

  // deserialize engine
  engine_handle_t engine = deserializeCudaEngine(runtime, engine_file_name);
  if (engine < 0) {
    fprintf(stderr, "unable to deserialize engine\n");
    goto destroy_runtime;
  }

  // get binding indices for input and output layers
  const char * input_name = "conv2d_input";
  const char * output_name = "dense_2/Softmax";
  int input_index = getBindingIndex(engine, input_name);
  int output_index = getBindingIndex(engine, output_name);
  printf("input  layer %s has binding index %d\n", input_name, input_index);
  printf("output layer %s has binding index %d\n", output_name, output_index);
  int max_batch_size = getMaxBatchSize(engine);
  printf("max batch size: %d\n", max_batch_size);

  // create execution context
  context_handle_t context = createExecutionContext(engine);
  if (context < 0) {
    fprintf(stderr, "unable to create execution context\n");
    goto destroy_engine;
  }

  // read image file
  int width, height;
  float *image_buf = read_pgm_file(image_file_name, &width, &height);
  if (image_buf == NULL) {
    goto destroy_context;
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      printf("%c", image_buf[y * width + x] < 0.5 ? '@' : ' ');
    }
    printf("\n");
  }

  // allocate device memory
  const int num_input_elements = width * height;
  float *dev_input, *dev_output;
  CUDA_TRY(cudaMalloc((void**)&dev_input, sizeof(float) * num_input_elements));
  CUDA_TRY(cudaMalloc((void**)&dev_output, sizeof(float) * NUM_OUTPUT_ELEMENTS));
  void *buffers[2];
  buffers[input_index] = dev_input;
  buffers[output_index] = dev_output;

  // copy image to device
  CUDA_TRY(cudaMemcpy(dev_input, image_buf, sizeof(float) * num_input_elements, cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemset(dev_output, 0, sizeof(float) * NUM_OUTPUT_ELEMENTS));
  float logits[NUM_OUTPUT_ELEMENTS];

  cudaStream_t stream;
  CUDA_TRY(cudaStreamCreate(&stream));
  const int batch_size = 1;
  enqueue(engine, batch_size, buffers, stream, NULL);
  CUDA_TRY(cudaStreamSynchronize(stream));

  CUDA_TRY(cudaMemcpy(logits, dev_output, NUM_OUTPUT_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
  float max_val = logits[0];
  int max_arg = 0;
  for (int i = 0; i < NUM_OUTPUT_ELEMENTS; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_arg = i;
    }
    printf("%d %.5f\n", i, logits[i]);
  }
  printf("prediction: %d\n", max_arg);

  return_code = 0;

  CUDA_TRY(cudaFree(dev_output));
  CUDA_TRY(cudaFree(dev_input));
  CUDA_TRY(cudaStreamDestroy(stream));


destroy_context:
  res = destroyExecutionContext(context);
  if (res < 0) {
    fprintf(stderr, "error while destroy execution context\n");
    return_code = -1;
  }

destroy_engine:
  res = destroyEngine(engine);
  if (res < 0) {
    fprintf(stderr, "error while destroying engine\n");
    return_code = -1;
    goto destroy_runtime;
  }

destroy_runtime:
  printf("destroying inference runtime...\n");
  res = destroyInferRuntime(runtime);
  if (res < 0) {
    fprintf(stderr, "error while destorying inference runtime\n");
    return_code = -1;
  }
  return return_code;
}
