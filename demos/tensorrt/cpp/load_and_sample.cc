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

#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <NvInfer.h>


#define CUDA_TRY(status) cudaCall_(status, __FILE__, __LINE__)

inline void cudaCall_(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::stringstream ss;
    ss << file << ':' << line << ": " << cudaGetErrorString(err) << std::endl;
    std::fputs(ss.str().c_str(), stderr);
    std::exit(EXIT_FAILURE);
  }
}


using DataType = nvinfer1::DataType;

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) override {
    //if (severity != Severity::kINFO) {
      std::cout << msg << std::endl;
    //}
  }
};
Logger gLogger{};


int get_num_elements_in_tensor(const nvinfer1::ICudaEngine *engine, int binding_index) {
  int count = 1;
  auto dims = engine->getBindingDimensions(binding_index);
  for (auto dim = 0 ; dim < dims.nbDims; ++dim) {
    count *= dims.d[dim];
  }
  return count;
}

struct Image {
 int width;
 int height;
 std::vector<float> image;

 int num_elements() const {
   return width * height;
 }
};

void read_pgm_file(Image *img, const std::string file_name) {
  std::ifstream image_file(file_name, std::ios::in);
  if (!image_file) {
    std::cerr << "unable to open file " << file_name << '\n';
    return;
  }
  img->width = 0;
  img->height = 0;
  std::string magic;
  int max_gray_level;
  image_file >> magic;
  image_file >> img->width >> img->height >> max_gray_level;
  auto offset = image_file.tellg();
  offset += 1; // skip line break after max_gray_level
  image_file.seekg(0, std::ios::end);
  const auto buffer_size_bytes = image_file.tellg() - offset;

  bool uses_uint16 = max_gray_level > 255;
  const int expected_buffer_size = img->width * img->height * (uses_uint16 ? sizeof(uint16_t) : sizeof(uint8_t));
  if (buffer_size_bytes != expected_buffer_size) {
    std::cerr << "file size does not match!\n";
    std::cerr << "width:" << img->width << ", height:" << img->height << ", max level:" << max_gray_level << std::endl;
    std::cerr << "expected buffer size: " << expected_buffer_size << " bytes\n";
    std::cerr << "actual buffer size: " << buffer_size_bytes << " bytes\n";
    return;
  }
  image_file.seekg(offset, std::ios::beg);
  img->image.reserve(img->width * img->height);
  if (uses_uint16) {
    std::vector<uint16_t> buf(buffer_size_bytes / 2, 0);
    int c = 0;
    for (auto u16 : buf) {
      std::cout << (u16 > max_gray_level / 2 ? '@' : ' ');
      c += 1;
      if (c % img->width == 0) {
        std::cout << '\n';
      }
    }
    image_file.read(reinterpret_cast<char*>(buf.data()), buffer_size_bytes);
    // invert image and scale to [0,1]
    std::transform(buf.begin(), buf.end(), std::back_inserter(img->image),
                   [max_gray_level](uint16_t u16) -> float { return 1.0f - static_cast<float>(u16) / max_gray_level; });
  } else {
    std::vector<uint8_t> buf(buffer_size_bytes, 0);
    image_file.read(reinterpret_cast<char*>(buf.data()), buffer_size_bytes);
    int c = 0;
    for (auto u8 : buf) {
      std::cout << (u8 > max_gray_level / 2 ? '@' : ' ');
      c += 1;
      if (c % img->width == 0) {
        std::cout << '\n';
      }
    }
    // invert image and scale to [0,1]
    std::transform(buf.begin(), buf.end(), std::back_inserter(img->image),
                   [max_gray_level](uint8_t u8) -> float { return 1.0f - static_cast<float>(u8) / max_gray_level; });

  }
  image_file.close();
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << argv[0] << " plan_file image.pgm\n";
    return -1;
  }
  std::string file_name(argv[1]);
  std::string image_file(argv[2]);
  std::ifstream plan_file(file_name, std::ios::in | std::ios::binary);
  if (!plan_file) {
    std::cerr << "unable to open file " << file_name << '\n';
    return -1;
  }

  plan_file.seekg(0, std::ios::end);
  std::size_t length = plan_file.tellg();
  plan_file.seekg(0, std::ios::beg);
  std::cout << "Engine " << file_name << ", " << length << " bytes\n";

  std::unique_ptr<char[]> engine_buffer(new char[length]);
  plan_file.read(engine_buffer.get(), length);
  plan_file.close();

  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
  nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_buffer.get(), length, nullptr);
  if (engine == nullptr) {
    std::cerr << "engine could not be deserialized\n";
    return -1;
  }
  engine_buffer.reset();

  nvinfer1::IExecutionContext *execution_context = engine->createExecutionContext();
  std::cout << "max batch size: " << engine->getMaxBatchSize() << std::endl;
  for (auto binding_index = 0; binding_index < engine->getNbBindings(); ++binding_index) {
    auto dims = engine->getBindingDimensions(binding_index);
    std::cout << binding_index << ": " << engine->getBindingName(binding_index)
      << " is_input:" << (engine->bindingIsInput(binding_index)?"yes":"no")
      << " dims:";
    for (auto dim = 0 ; dim < dims.nbDims; ++dim) {
      if (dim == 0) {
        std::cout << "[";
      } else {
        std::cout << ", ";
      }
      std::cout << dims.d[dim];
    }
    std::cout << "], dtype: ";
    switch(engine->getBindingDataType(binding_index)) {
      case DataType::kFLOAT:
        std::cout << "kFloat";
        break;
      case DataType::kHALF:
        std::cout << "kHalf";
        break;
      case DataType::kINT8:
        std::cout << "quantized INT8";
        break;
      case DataType::kINT32:
        std::cout << "INT32";
        break;
      default:
        std::cout << "unknown";
    }
    std::cout << ", format:" << engine->getBindingFormatDesc(binding_index) << std::endl;
  }
  const char* input_name = "conv2d_input";
  const char* output_name = "dense_2/Softmax";
  int input_index = engine->getBindingIndex(input_name);
  int output_index = engine->getBindingIndex(output_name);
  std::cout << "input index: " << input_index << std::endl;
  std::cout << "output index: " << output_index << std::endl;

  const int num_input_elements = get_num_elements_in_tensor(engine, input_index);
  const int num_output_elements = get_num_elements_in_tensor(engine, output_index);
  std::cout << "input tensor:  " << num_input_elements << " elements\n";
  std::cout << "output tensor: " << num_output_elements << " elements\n";

  cudaStream_t stream;
  CUDA_TRY(cudaStreamCreate(&stream));

  // allocate device memory
  float *dev_input; float *dev_output;
  CUDA_TRY(cudaMalloc(&dev_input, sizeof(float) * num_input_elements));
  CUDA_TRY(cudaMalloc(&dev_output, sizeof(float) * num_output_elements));
  void* buffers[2];
  buffers[input_index] = dev_input;
  buffers[output_index] = dev_output;

  // load image
  Image image{};
  read_pgm_file(&image, image_file);
  if (num_input_elements != image.num_elements()) {
    std::cerr << "expected image with " << num_input_elements << " pixels but got "
              << image.num_elements() << std::endl;
    return -1;
  }
  // copy image to device
  CUDA_TRY(cudaMemcpy(dev_input, image.image.data(), sizeof(float) * image.num_elements(),
                      cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemset(dev_output, 0, sizeof(float) * num_output_elements));
  std::vector<float> logits(num_input_elements);

  constexpr int batch_size = 1;
  execution_context->enqueue(batch_size, buffers, stream, nullptr);
  CUDA_TRY(cudaStreamSynchronize(stream));

  CUDA_TRY(cudaMemcpy(logits.data(), dev_output, num_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_output_elements; ++i) {
    std::cout << i << ": " << logits[i] << std::endl;
  }
  auto pred = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
  std::cout << "prediction: " << pred << std::endl;

  CUDA_TRY(cudaFree(dev_output));
  CUDA_TRY(cudaFree(dev_input));
  CUDA_TRY(cudaStreamDestroy(stream));
  execution_context->destroy();
  engine->destroy();
  runtime->destroy();
}
