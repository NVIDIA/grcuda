# GrCUDA and TensorRT

This example uses TensorFlow 1.x to train a LeNet5 model on the MNIST dataset. An inference engine is then created
from the trained module using TensorRT through its Python API. This engine is serialized to a file.
The engine is subsequently instantiated from a Node.js application using GrCUDA.

The serialization of the frozen model to a Protobuf file is only supported in TensorFlow 1.x.
As per Tensor-RT 7.0, its provided end-to-end examples are only working under TensorFlow 1.x.

## Model Training in TensorFlow 1.x

[`model.py`](python/model.py) defines and trains LeNet5 model on the
MNIST dataset.

```console
$ cd python
$ python model.py
using TensorFlow version:  1.14.0
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 6)         60
_________________________________________________________________
average_pooling2d (AveragePo (None, 13, 13, 6)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 16)        880
_________________________________________________________________
average_pooling2d_1 (Average (None, 5, 5, 16)          0
_________________________________________________________________
flatten (Flatten)            (None, 400)               0
_________________________________________________________________
dense (Dense)                (None, 120)               48120
_________________________________________________________________
dense_1 (Dense)              (None, 84)                10164
_________________________________________________________________
dense_2 (Dense)              (None, 10)                850
=================================================================
Total params: 60,074
Trainable params: 60,074
Non-trainable params: 0
_________________________________________________________________
...
```

This generates a Protobuf file `model/lenet5.pb` from the trained model.

## Generation of Inference Engine

In a first step, the `model/lenet5.pb` model must be converted to a UFF model (universal framework format).

```console
$ convert-to-uff models/lenet5.pb
$ ls models/
lenet5.pb  lenet5.uff
```

In the second, step the inference engine is created and serialized to a file. This is done by the
[`build_engine.py`](python/build_engine.py) script. The parameters used TensorRT are specified in this Python script.

```console
$ cd python
$ python build_engine.py ../models/lenet5.uff ../models/lenet5.engine
$ ls ../models/
lenet5.engine  lenet5.pb  lenet5.uff
```

## Download MNIST Test Images

Run the script `download_mnist_test_digits.py` in the `data` directory to download the
MNIST data set and extract the first occurrence of the digits zero to nine of the
test set. This script stores the images in PGM format as `0.pgm` to `9.pgm`.

```console
$ cd data
$ python download_mnist_test_digits.py
writing 0.pgm...
writing 1.pgm...
writing 2.pgm...
writing 3.pgm...
writing 4.pgm...
writing 5.pgm...
writing 6.pgm...
writing 7.pgm...
writing 8.pgm...
writing 9.pgm...
```

## Instantiation of Inference Engine from CPython

To test the generated engine, the [`load_and_sample.py`](python/load_and_sample.py)
script can be used. It instantiates the engine from
the file and runs a sample image through the network.

```console
$ python load_and_sample.py ../models/lenet5.engine ../data/8.pgm
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@  @@@@@@
@@@@@@@@@@@@@@@@@@@@  @@@@@@
@@@@@@@@@@@@          @@@@@@
@@@@@@@@@@@@         @@@@@@@
@@@@@@@@@@     @@@   @@@@@@@
@@@@@@@@@@   @@@@   @@@@@@@@
@@@@@@@@@@@   @@   @@@@@@@@@
@@@@@@@@@@@   @   @@@@@@@@@@
@@@@@@@@@@@       @@@@@@@@@@
@@@@@@@@@@@@     @@@@@@@@@@@
@@@@@@@@@@@     @@@@@@@@@@@@
@@@@@@@@@@      @@@@@@@@@@@@
@@@@@@@@@@      @@@@@@@@@@@@
@@@@@@@@@  @@@  @@@@@@@@@@@@
@@@@@@@@  @@@  @@@@@@@@@@@@@
@@@@@@@   @@@  @@@@@@@@@@@@@
@@@@@@@   @@   @@@@@@@@@@@@@
@@@@@@@      @@@@@@@@@@@@@@@
@@@@@@@@     @@@@@@@@@@@@@@@
@@@@@@@@@  @@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
0: 0.000015
1: 0.000000
2: 0.000044
3: 0.000007
4: 0.000000
5: 0.000003
6: 0.000000
7: 0.000000
8: 0.999812
9: 0.000120
Prediction: 8
```

## Native Instantiation of Inference Engine (optional)

Now, we show how to instantiate the serialized engine in a native
C++ application (`cpp/load_and_sample.cc`) using the C++ TensorRT
inference library.
This is an optional step provided here only for completeness. It
does not use GrCUDA C-wrapper library `libtrt.so` nor GrCUDA.

```console
$ cd cpp
$ mkdir build
$ cd build
$ cmake .. -DTENSORRT_DIR=/usr/local/TensorRT-7.0.0.11/
$ make
...

$ ./load_and_sample ../../models/lenet5.engine ../../data/4.pgm
Engine ../../models/lenet5.engine, 252822 bytes
Deserialize required 1144468 microseconds.
max batch size: 1
0: conv2d_input is_input:yes dims:[1, 28, 28], dtype: kFloat, format:Row major linear FP32 format (kLINEAR)
1: dense_2/Softmax is_input:no dims:[10, 1, 1], dtype: kFloat, format:Row major linear FP32 format (kLINEAR)
input index: 0
output index: 1
input tensor:  784 elements
output tensor: 10 elements
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
0: 9.16423e-07
1: 7.97255e-06
2: 2.85043e-05
3: 1.39731e-05
4: 0.992048
5: 8.05108e-07
6: 1.88485e-07
7: 5.72944e-05
8: 6.37524e-05
9: 0.00777862

```

## Instantiation of Inference Engine from GrCUDA

First, build that the GrCUDA wrapper library `libtrt` for TensorRT.

```console
$ cd <GrCUDA repo root>../tensorrt
$ mkdir build
$ cd build
$ cmake .. -DTENSORRT_DIR=/usr/local/TensorRT-7.0.0.11/
$ make -j
```

You can validate the `libtrt` wrapper library using the C test
application in [`libtrt_load_and_sample.c`](../../tensorrt/app/libtrt_load_and_sample.c).

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

Finally, instantiate the TensorRT engine from a Node.js program [`tensorrt_example`](tensorrt_example.js).

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
