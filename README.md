# grCUDA: Polyglot GPU Access in GraalVM

This Truffle language exposes GPUs to the polyglot [GraalVM](http://www.graalvm.org). The goal is to

1) make data exchange between the host language and the GPU efficient without burdening the programmer.

2) allow programmers to invoke _existing_ GPU kernels from their host language.

Supported and tested GraalVM languages:

- Python
- JavaScript/NodeJS
- Ruby
- R
- Java
- C and Rust through the Graal Sulong Component

For details and features of grCUDA language see the [grCUDA documentation](docs/language.md).

How to bind precompiled kernels to callables, compile and launch kernels is
described in the [polyglot kernel launch](docs/launchkernel.md) documentation.

## Using grCUDA in the GraalVM

grCUDA can be used in the binaries of the GraalVM languages (`lli`, `graalpython`,
`js`, `R`, and `ruby)`. The JAR file containing grCUDA must be appended to the classpath
or copied into `jre/languages/grcuda` of the Graal installation. Note that `--jvm`
and `--polyglot` must be specified in both cases as well.

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

```console
$GRAALVM_DIR/bin/node --polyglot --jvm example.js
1
2
...
100
```

The next example shows how to launch an existing compiled GPU kernel from Python.
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

is compiled using `nvcc --cubin` into a cubin file. The kernel function can be loaded from the cubin and bound to a callable object in the host language, here Python.

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

```console
nvcc --cubin  --generate-code arch=compute_70,code=sm_70 kernel.cu
$GRAALVM_DIR/bin/graalpython --polyglot --jvm example.py
1
2
...
100
```

For more details on how to invoke existing GPU kernels, see the
Documentation on [polyglot kernel launches](docs/launchkernel.md).

## Installation

grCUDA can be downloaded as a binary JAR from [grcuda/releases](https://github.com/NVIDIA/grcuda/releases) and manually copied into a GraalVM installation.

1. Download GraalVM CE 20.0.0 for Linux `graalvm-ce-java8-linux-amd64-20.0.0.tar.gz`
   from [GitHub](https://github.com/oracle/graal/releases) and untar it in your
   installation directory.

   ```console
   cd <your installation directory>
   tar xfz graalvm-ce-java8-linux-amd64-20.0.0.tar.gz
   export GRAALVM_DIR=`pwd`/graalvm-ce-java8-20.0.0
   ```

2. Download the grCUDA JAR from [grcuda/releases](https://github.com/NVIDIA/grcuda/releases)

   ```console
   cd $GRAALVM_DIR/jre/languages
   mkdir grcuda
   cp <download folder>/grcuda-0.1.0.jar grcuda
   ```

3. Test grCUDA in Node.JS from GraalVM.

   ```console
   cd $GRAALVM_DIR/bin
   ./node --jvm --polyglot
   > arr = Polyglot.eval('grcuda', 'int[5]')
   [Array: null prototype] [ 0, 0, 0, 0, 0 ]
   ```

4. Download other GraalVM languages.

   ```console
   cd $GRAAL_VM/bin
   ./gu available
   ./gu install python
   ./gu install R
   ./gu install ruby
   ```

## Instructions to build grCUDA from Sources

grCUDA requires the [mx build tool](https://github.com/graalvm/mx). Clone the mx reposistory and
add the directory into `$PATH`, such that the `mx` can be invoked from the command line.

Build grCUDA and the unit tests:

```console
cd <directory containing this REAMDE>
mx build
```

Note that this will also checkout the graal repository.

To run unit tests:

```bash
mx unittest com.nvidia
```

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
