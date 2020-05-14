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

#ifndef TRT_H
#define TRT_H

#ifdef __cplusplus
extern "C" {
#endif

//
// libtrt is a C-style wrapper for the most basic functions of TensorRT.
//

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