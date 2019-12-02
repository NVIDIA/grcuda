#include <atomic>
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
    //if (severity != Severity::kINFO) {
      std::cout << msg << std::endl;
    //}
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

engine_handle_t deserializeCudaEngine(runtime_handle_t handle, void* buffer, size_t num_bytes) {
  auto runtime = runtimeFromHandle(handle);
  if (runtime == nullptr) {
    return TRT_INVALID_HANDLE;
  }
  auto engine = runtime->deserializeCudaEngine(buffer, num_bytes, nullptr);
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
