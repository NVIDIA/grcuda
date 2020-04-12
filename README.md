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

The following example shows how to create a GPU kernel and two device arrays
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

## Step-by-step development guide

* This section contains all the steps required to setup GrCUDA if your goal is to contribute to its development, or simply hack with it. This guide refers to GraalVM Community Edition JDK8 for Linux with `amd64` architectures, i.e. download releases prefixed with `graalvm-ce-java8-linux-amd64` or something like that. 

1. Get the source code of GrCUDA, graal, mx

	```
	git clone https://github.com/oracle/graal.git
	git clone https://github.com/graalvm/mx.git
	git clone https://github.com/NVIDIA/grcuda.git (this can be replaced with a fork)
	```

2. Download the right JDK
	* [Here](https://github.com/graalvm/openjdk8-jvmci-builder/releases/tag/jvmci-20.0-b02) you can find releases for GraalVM 20.0, but other versions are available on the same repository

3. Download the right build for GraalVM
	* [Here](https://github.com/graalvm/graalvm-ce-builds/releases) you can find releases for GraalVM 20.0, and more recent versions once they will become available

4. Setup your CUDA environment
    * Install CUDA and Nvidia drivers, for exampel following the steps [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=rpmnetwork)
	* Add the following to your environment (assuming that you have installed CUDA in the default `/usr/local` location, and using the `nvcc` compiler

	```
	export CUDA_DIR=/usr/local/cuda
	export PATH=$PATH:$CUDA_DIR/bin
	```

5. Setup your GraalVM and GrCUDA environment
	* Add the following to your environment (assuming you have installed the releases mentioned in step 2 and 3)

	```
	export PATH=~/mx:$PATH
	export JAVA_HOME=~/openjdk1.8.0_242-jvmci-20.0-b02
	export GRAAL_HOME=~/graalvm-ce-java8-20.0.0
	export GRAALVM_HOME=$GRAAL_HOME
	export PATH=$GRAAL_HOME/bin:$PATH
	export PATH=$JAVA_HOME/bin:$PATH
	```

6. Install languages for GraalVM (optional, but recommended)
	```
	gu available
    gu install native-image
	gu install llvm-toolchain
	gu install python 
    gu rebuild-images polyglot
	```

	* If Graalpython is installed, create a `virtualenv` for it

	```
    graalpython -m venv ~/graalpython_venv
    source ~/graalpython_venv/bin/activate
	```

7. Install GrCUDA with `./install.sh`

8. Setup your IDE with `mx ideinit`

9. Run tests with `mx unittest com.nvidia`
	



