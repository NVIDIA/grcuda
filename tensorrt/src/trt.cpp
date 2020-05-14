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

#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <NvInfer.h>
#include <trt.h>

using RuntimeHandleMap = std::unordered_map<runtime_handle_t, nvinfer1::IRuntime*>;
using EngineHandleMap = std::unordered_map<engine_handle_t, nvinfer1::ICudaEngine*>;
using ContextHandleMap = std::unordered_map<context_handle_t, nvinfer1::IExecutionContext*>;

static RuntimeHandleMap runtime_handle_map{};
static EngineHandleMap engine_handle_map{};
static ContextHandleMap context_handle_map{};
static std::atomic_int next_runtime_handle{1};
static std::atomic_int next_engine_handle{1};
static std::atomic_int next_context_handle{1};
static std::mutex monitor;

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) override {
    if (severity != Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
};
static Logger logger{};


nvinfer1::IRuntime* runtimeFromHandle(runtime_handle_t handle) {
  std::lock_guard<std::mutex> lock(monitor);
  auto it = runtime_handle_map.find(handle);
  if (it == runtime_handle_map.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

nvinfer1::ICudaEngine* engineFromHandle(engine_handle_t handle) {
  std::lock_guard<std::mutex> lock(monitor);
  auto it = engine_handle_map.find(handle);
  if (it == engine_handle_map.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

nvinfer1::IExecutionContext* contextFromHandle(context_handle_t handle) {
  std::lock_guard<std::mutex> lock(monitor);
  auto it = context_handle_map.find(handle);
  if (it == context_handle_map.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

runtime_handle_t createInferRuntime() {
  auto handle = next_runtime_handle++;
  auto runtime = nvinfer1::createInferRuntime(logger);
  if (runtime == nullptr) {
    return TRT_UNABLE_TO_CREATE_RUNTIME;
  }
  {
    std::lock_guard<std::mutex> lock(monitor);
    runtime_handle_map.emplace(handle, runtime);
  }
  return handle;
}

engine_handle_t deserializeCudaEngine(runtime_handle_t handle, const char *engine_file_name) {
  auto runtime = runtimeFromHandle(handle);
  if (runtime == nullptr) {
    return TRT_INVALID_HANDLE;
  }
  std::ifstream plan_file(engine_file_name, std::ios::in | std::ios::binary);
  if (!plan_file) {
    return TRT_ENGINE_FILE_NOT_FOUND;
  }
  plan_file.seekg(0, std::ios::end);
  std::size_t length = plan_file.tellg();
  plan_file.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> engine_buffer(new char[length]);
  plan_file.read(engine_buffer.get(), length);
  plan_file.close();
  auto engine = runtime->deserializeCudaEngine(engine_buffer.get(), length, nullptr);
  if (engine == nullptr) {
    return TRT_ENGINE_DESERIALIZATION_ERROR;
  }
  auto engine_handle = next_engine_handle++;
  {
    std::lock_guard<std::mutex> lock(monitor);
    engine_handle_map.emplace(engine_handle, engine);
  }
  return engine_handle;
}

context_handle_t createExecutionContext(engine_handle_t handle) {
  auto engine = engineFromHandle(handle);
  if (engine == nullptr) {
    return TRT_INVALID_HANDLE;
  }
  auto context = engine->createExecutionContext();
  auto context_handle = next_context_handle++;
  {
    std::lock_guard<std::mutex> lock(monitor);
    context_handle_map.emplace(context_handle, context);
  }
  return context_handle;
}

int getBindingIndex(engine_handle_t handle, const char *name) {
  auto engine = engineFromHandle(handle);
  if (engine == nullptr) {
    return TRT_INVALID_HANDLE;
  }
  return engine->getBindingIndex(name);
}

int getMaxBatchSize(engine_handle_t handle) {
  auto engine = engineFromHandle(handle);
  if (engine == nullptr) {
    return TRT_INVALID_HANDLE;
  }
  return engine->getMaxBatchSize();
}

trt_error_t destroyEngine(engine_handle_t handle) {
  auto engine = engineFromHandle(handle);
  if (engine == nullptr) {
    return TRT_INVALID_HANDLE;
  }
  engine->destroy();
  {
    std::lock_guard<std::mutex> lock(monitor);
    engine_handle_map.erase(handle);
  }
  return TRT_OK;
}

int enqueue(context_handle_t handle, int batch_size, void **bindings, cudaStream_t stream, cudaEvent_t *input_consumed) {
  auto context = contextFromHandle(handle);
  if (context == nullptr) {
    return 0;
  }
  return context->enqueue(batch_size, bindings, stream, input_consumed);
}

trt_error_t destroyExecutionContext(context_handle_t handle) {
  auto context = contextFromHandle(handle);
  if (context == nullptr) {
    return TRT_INVALID_HANDLE;
  }
  context->destroy();
  {
    std::lock_guard<std::mutex> lock(monitor);
    context_handle_map.erase(handle);
  }
  return TRT_OK;
}

trt_error_t destroyInferRuntime(runtime_handle_t handle) {
  auto runtime = runtimeFromHandle(handle);
  if (runtime == nullptr) {
    return TRT_INVALID_HANDLE;
  }
  runtime->destroy();
  {
    std::lock_guard<std::mutex> lock(monitor);
    runtime_handle_map.erase(handle);
  }
  return TRT_OK;
}
