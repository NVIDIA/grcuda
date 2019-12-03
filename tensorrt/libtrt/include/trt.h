#ifndef TRT_H
#define TRT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int runtime_handle_t;
typedef int engine_handle_t;
typedef int context_handle_t;

enum trt_error {
  TRT_OK = 0,
  TRT_INVALID_HANDLE = -1,
  TRT_UNABLE_TO_CREATE_RUNTIME = -2,
  TRT_ENGINE_DESERIALIZATION_ERROR = -3,
  TRT_ENGINE_FILE_NOT_FOUND = -4
};
typedef enum trt_error trt_error_t;

// IRuntime
runtime_handle_t createInferRuntime();
engine_handle_t deserializeCudaEngine(runtime_handle_t handle, const char *engine_file_name);
trt_error_t destroyInferRuntime(runtime_handle_t handle);

// ICudaEngine
context_handle_t createExecutionContext(engine_handle_t handle);
int getBindingIndex(engine_handle_t handle, const char *name);
int getMaxBatchSize(engine_handle_t handle);
trt_error_t destroyEngine(engine_handle_t handle);

// IExecutionContext
int enqueue(context_handle_t handle, int batch_size, void **bindings, cudaStream_t stream, cudaEvent_t *input_consumed);
trt_error_t destroyExecutionContext(context_handle_t handle);



#ifdef __cplusplus
}
#endif

#endif // TRT_H