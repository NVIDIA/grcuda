# grCUDA: Polyglot GPU Access

This Tuffle language exposes GPUs to the polyglot [GraalVM](http://www.graalvm.org). The goal is to

1) make data exchange between the host language and the GPU efficient without burdening the programmer.

2) allow programmers to invoke _existing_ GPU kernels from their host language.

Supported and tested host languages:

- Python
- JavaScript/NodeJS
- Ruby
- R
- Java
- C and Rust through the Graal Sulong Component

For details and features of grCUDA language see
the [grCUDA documentation](docs/language.md).

Binding and launching of precompiled kernels is described in the
[polyglot kernel launch](docs/launchkernel.md) documentation.

## Using grCUDA in the GraalVM

grCUDA can be used in [GraalVM binaries](https://www.graalvm.org/downloads/)
(`lli`, `graalpython`, `js`, `R`, and `ruby`). The JAR file containing the
grCUDA must be appended to the class path. Note that `--jvm` and
`--polyglot` must be specified too.

The following example shows how create a GPU kernel and two device arrays
in JavaScript (NodeJS) and invoke the kernel:

```JavaScript
// build kernel from CUDA C/C++ source code
const kernelSource = `
__global__ void increment(int *arr, int n) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    arr[idx] += 1;
  }
}`
const buildkernel = Polyglot.eval('grcuda', 'buildkernel')
const incKernel = buildkernel(
  kernelSource, // CUDA kernel source code string
  'increment', // kernel name
  'pointer, sint32') // kernel signature

// allocate device array
const n = 100
const deviceArray = Polyglot.eval('grcuda', 'int[100]')
for (let i = 0; i < n; i++) {
  deviceArray[i] = i // ... and initialize on the host
}
// launch kernel in grid of 1 block with 128 threads
incKernel(1, 128)(deviceArray, n)

// print elements from updated array
for (let i = 0; i < n; ++i) {
  console.log(deviceArray[i])
}
```

```bash
$SOME_DIR/graalvm-ce-19.0.0/bin/node --polyglot --jvm \
  --vm.Dtruffle.class.path.append=`pwd`/mxbuild/dists/jdk1.8/grcuda.jar \
  example.js
1
2
...
100
```

The next example shows how to launch an existing compiled GPU kernel from JavaScript.
The CUDA kernel

```C
extern "C"
__global__ void increment(int *arr, int n) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    arr[idx] += 1;
  }
}
```

is compiled using `nvcc --cubin` into a cubin binary. The kernel function can be loaded from this cubin and bound to a callable object in the host language, here JavaScript.

```Python
import polyglot

n = 100
device_array = polyglot.eval(language='grcuda', string='int[100]')
for i in range(n):
  device_array[i] = i

inc_kernel = polyglot.eval(   # bind kernel from binary
  language='cuda',
  string='bindkernel("kernel.cubin", "increment", "pointer, sint32")')

# launch kernel as 1 block with 128 threads
inc_kernel(1, 128)(device_array, n)

for i in range(n):
  print(device_array[i])
```

```bash
nvcc --cubin  --generate-code arch=compute_70,code=sm_70 kernel.cu
$SOME_DIR/graalvm-ce-19.0.0/bin/graalpython --polyglot --jvm \
  --vm.Dtruffle.class.path.append=`pwd`/mxbuild/dists/jdk1.8/grcuda.jar \
  example.py
1
2
...
100
```

For more details on how to invoke existing GPU kernels, see the
Documentation on [polyglot kernel launches](docs/launchkernel.md).

## Using grCUDA in a JDK

Make sure that you use the [OpenJDK+JVMCI-0.55](https://github.com/graalvm/openjdk8-jvmci-builder/releases/tag/jvmci-0.55).

To use the CUDA language from Python:

```text
mx --dynamicimports graalpython --cp-sfx `pwd`/mxbuild/dists/jdk1.8/grcuda.jar \
   python --polyglot
...
>>> import polyglot
>>> da = polyglot.eval(language='grcuda', string='double[1000]')
>>> da[0]
0.0
>>> da[0] = 1.2
>>> da[0:10]
[1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

## Build Instructions

To build the language and the unit tests:

```bash
mx build
```

To run unit tests:

```bash
mx unittest com.nvidia
```
