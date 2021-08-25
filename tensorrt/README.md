# GrCUDA and TensorRT

This directory contains a wrapper library `libtrt.so` for TensorRT.
It simplifies the use of the TensorRT inference library.

## Build libtrt

Build that the GrCUDA wrapper library `libtrt` for TensorRT.

```console
$ cd <GrCUDA repo root>../tensorrt
$ mkdir build
$ cd build
$ cmake .. -DTENSORRT_DIR=/usr/local/TensorRT-7.0.0.11/
$ make -j
```

You can validate the `libtrt` wrapper library using the C test
application in `app/libtrt_load_and_sample.c`.

If not already downloaded, download some test images from the MNIST data set
into the `data` directory. Run the script `download_mnist_test_digits.py` in the
`data` directory to download the MNIST data set and extract the first occurrence
of the digits zero to nine in the test set.
This script stores the images in PGM format as `0.pgm` to `9.pgm`.

```console
$ cd tensorrt/build
$ ./libtrt_load_and_sample ../../examples/tensorrt/models/lenet5.engine \
                           ../../examples/tensorrt/data/4.pgm
creating inference runtime...
Deserialize required 1135802 microseconds.
input  layer conv2d_input has binding index 0
output layer dense_2/Softmax has binding index 1
max batch size: 1
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@ @@@@@@
@@@@@@@@@@@@@@@@@@@@@ @@@@@@
@@@@@ @@@@@@@@@@@@@@  @@@@@@
@@@@  @@@@@@@@@@@@@@  @@@@@@
@@@@  @@@@@@@@@@@@@  @@@@@@@
@@@@  @@@@@@@@@@@@@  @@@@@@@
@@@@  @@@@@@@@@@@@@  @@@@@@@
@@@  @@@@@@@@@@@@@   @@@@@@@
@@@  @@@@@@@@@@@@   @@@@@@@@
@@@  @@@@@@@        @@@@@@@@
@@@            @@@  @@@@@@@@
@@@@@     @@@@@@@@  @@@@@@@@
@@@@@@@@@@@@@@@@@   @@@@@@@@
@@@@@@@@@@@@@@@@@  @@@@@@@@@
@@@@@@@@@@@@@@@@@  @@@@@@@@@
@@@@@@@@@@@@@@@@@  @@@@@@@@@
@@@@@@@@@@@@@@@@@  @@@@@@@@@
@@@@@@@@@@@@@@@@@   @@@@@@@@
@@@@@@@@@@@@@@@@@   @@@@@@@@
@@@@@@@@@@@@@@@@@@  @@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
0 0.00000
1 0.00001
2 0.00003
3 0.00001
4 0.99205
5 0.00000
6 0.00000
7 0.00006
8 0.00006
9 0.00778
prediction: 4
destroying inference runtime...
```

## Use libtrt in GrCUDA

```bash
GRCUDA_JAR="$GRCUDA_BUILD_DIR/mxbuild/dists/jdk1.8/grcuda.jar"
LIBTRT="$GRCUDA_BUILD_DIR/tensorrt/libtrt/build/libtrt.so"
LD_LIBRARY_PATH="/usr/local/TensorRT-7.0.0.11/lib:$LD_LIBRARY_PATH"

${GRAALVM_DIR}/bin/node --polyglot --jvm \
  --grcuda.TensorRTEnabled=true \
  --grcuda.TensorRTLibrary=$LIBTRT \
  --vm.Dtruffle.class.path.append=$GRCUDA_JAR \
  tensorrt_example.js data/4.pgm
```

A complete example for Node.js is provided in the [examples directory](../examples/tensorrt/README.md).

## Functions in librtr

```text
// Inference Runtime
createInferRuntime(): sint32      // returns runtime handle
destroyInferRuntime(rt_handle: sint32): void

// Engine
deserializeCudaEngine(rt_handle: sint32, engine_file_name: string): sint32
                      // returns engine handle
getBindingIndex(engine_handle: sint32, name: string): sint32
getMaxBatchSize(engine_handle: sint32): sint32
destroyEngine(engine_handle: sint32): void

// Execution Context
createExecutionContext(engine_handle: sint32): sint32
                       // returns context handle
destroyExecutionContext(context_handle: sint32): sint32

// Enqueue inference for batch
enqueue(context_handle: sint32, batch_size: sint32,
        buffers: arraylike): bool
                       // returns 'true' if successful
```

## Enqueue Inference Jobs

Inference jobs are executed asynchronous after being submitted
with `TRT::enqueue`. This function takes a bindings from a binding
index to the respective input or output buffers.
The bindings is specified in an array-like , e.g.,
an `Array` in JavaScript, and passed to `enqueue`.

Example in JavaScript:

```javascript
const imageDeviceArray = cu.DeviceArray(...)
const classProbabilitiesDeviceArray = cu.DeviceArray(...)

const inputIndex = trt.getBindingIndex(engine, 'conv2d_input')
const outputIndex = trt.getBindingIndex(engine, 'dense_2/Softmax')

const batchSize = 1
const buffers = []
buffers[inputIndex] = imageDeviceArray
buffers[outputIndex] = classProbabilitiesDeviceArray
trt.enqueue(engine, batchSize, buffers)
```
