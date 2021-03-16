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

A description of grCUDA and its the features can be found in the [grCUDA documentation](docs/grcuda.md).

The [bindings documentation](docs/bindings.md) contains a tutorial that shows
how to bind precompiled kernels to callables, compile and launch kernels.

**Additional Information:**

- [grCUDA: A Polyglot Language Binding for CUDA in GraalVM](https://devblogs.nvidia.com/grcuda-a-polyglot-language-binding-for-cuda-in-graalvm/). NVIDIA Developer Blog,
  November 2019.
- [grCUDA: A Polyglot Language Binding](https://youtu.be/_lI6ubnG9FY). Presentation at Oracle CodeOne 2019, September 2019.
- [Simplifying GPU Access](https://developer.nvidia.com/gtc/2020/video/s21269-vid). Presentation at NVIDIA GTC 2020, March 2020.

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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    arr[idx] += 1;
  }
}`
const cu = Polyglot.eval('grcuda', 'CU') // get grCUDA namespace object
const incKernel = cu.buildkernel(
  kernelSource, // CUDA kernel source code string
  'increment', // kernel name
  'pointer, sint32') // kernel signature

// allocate device array
const numElements = 100
const deviceArray = cu.DeviceArray('int', numElements)
for (let i = 0; i < numElements; i++) {
  deviceArray[i] = i // ... and initialize on the host
}
// launch kernel in grid of 1 block with 128 threads
incKernel(1, 128)(deviceArray, numElements)

// print elements from updated array
for (const element of deviceArray) {
  console.log(element)
}
```

```console
$GRAALVM_DIR/bin/node --polyglot --jvm example.js
1
2
...
100
```

### Calling existing compiled GPU Kernels

The next example shows how to launch an __existing compiled__ GPU kernel from Python.
The CUDA kernel

```C
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

num_elements = 100
cu = polyglot.eval(language='grcuda', string='CU')
device_array = cu.DeviceArray('int', num_elements)
for i in range(num_elements):
  device_array[i] = i

# bind to kernel from binary
inc_kernel = cu.bindkernel('kernel.cubin',
  'cxx increment(arr: inout pointer sint32, n: sint32)')

# launch kernel as 1 block with 128 threads
inc_kernel(1, 128)(device_array, num_elements)

for i in range(num_elements):
  print(device_array[i])
```

```console
nvcc --cubin  --generate-code arch=compute_75,code=sm_75 kernel.cu
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

grCUDA requires the [mx build tool](https://github.com/graalvm/mx). Clone the mx
repository and add the directory into `$PATH`, such that the `mx` can be invoked from
the command line.

Build grCUDA and the unit tests:

```console
cd <directory containing this README>
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

1. **Get the source code of GrCUDA, graal, mx**

```
git clone https://github.com/oracle/graal.git
git clone https://github.com/graalvm/mx.git
git clone https://github.com/NVIDIA/grcuda.git (this can be replaced with a fork)
```

2. **Download the right JDK**
* [Here](https://github.com/graalvm/graal-jvmci-8/releases) you can find releases for GraalVM 20.2 or newer, but other versions are available on the same repository

3. **Download the right build for GraalVM**
* [Here](https://github.com/graalvm/graalvm-ce-builds/releases) you can find releases for GraalVM 20.2, and more recent versions once they will become available

4. **Setup your CUDA environment**
* Install CUDA and Nvidia drivers, for example following the steps [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=rpmnetwork)
* Add the following to your environment (assuming you have installed CUDA in the default `/usr/local` location, and using the `nvcc` compiler

```
export CUDA_DIR=/usr/local/cuda
export PATH=$PATH:$CUDA_DIR/bin
```

5. **Setup your GraalVM and GrCUDA environment**
* Add the following to your environment (assuming you have installed the releases mentioned in step 2 and 3)

```
export PATH=~/mx:$PATH
export JAVA_HOME=~/openjdk1.8.0_242-jvmci-20.0-b02
export GRAAL_HOME=~/graalvm-ce-java8-20.0.0
export GRAALVM_HOME=$GRAAL_HOME
export PATH=$GRAAL_HOME/bin:$PATH
export PATH=$JAVA_HOME/bin:$PATH
```

6. **Install languages for GraalVM** (optional, but recommended)

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

* Recommended: install 'numpy` in Graalpython (required for performance benchmarking)

```
graalpython -m ginstall install setuptools;
graalpython -m ginstall install Cython;
graalpython -m ginstall install numpy;
```

7. **Install GrCUDA with** `./install.sh`

8. **Setup your IDE with** `mx ideinit`
* In IntelliJ Idea, install the Python plugin, then do Project Structure -> SDKs -> Create a new Python 2.7 Virtual Environment, it is used by `mx`
* In IntelliJ Idea, Project Structures -> Modules -> Set the Module SDK (under Dependencies) of `mx` and submodules to your Java SDK (e.g. `1.8`)
* Also update the project SDK and the default JUnit configurations to use the GraalVM SDK in `$GRAAL_HOME`, and update the `PATH` variable so that it can find `nvcc`
	* Modify the template Junit test configuration adding `-Djava.library.path="/path/to/graalvmbuild/lib` (in Java 11) or -Djava.library.path="graalvm-ce-java8-20.2.0/jre/lib/amd64"to the VM options to find `trufflenfi`
 and update the environment variables with `PATH=your/path/env/var` to find `nvcc`
	* In IntelliJ Idea, Run -> Edit Configurations, then create a new JUnit configuration set to "All in package" with `com.nvidia.grcuda` as module and `com.nvidia.grcuda.test` below, add to VM options `-Djava.library.path="/path/to/graalvm-ce-java8-20.0.0/jre/lib/amd64"` (or your version of GraalVM). Specify the SDK by setting the GraalVM JRE in e.g. `/path/to/graalvm-ce-java8-20.0.0`
	* Environment variables should have PATH identical to what you use in a shell
9. **Run tests with** `mx unittest com.nvidia`
* Run a specific test using, for example, `mx unittest com.nvidia.grcuda.test.gpu.ExecutionDAGTest#executionDAGConstructorTest`

10. **Add your GrCUDA directory to the environment with** `export GRCUDA_HOME=/path/to/grcuda`

11. **Execute performance tests using Graalpython**

Run a specific benchmark with custom settings
```
graalpython --jvm --polyglot --grcuda.InputPrefetch --grcuda.ForceStreamAttach --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.ExecutionPolicy=default --grcuda.DependencyPolicy=with-const --grcuda.RetrieveParentStreamPolicy=disjoint benchmark_main.py -d -i 10 -n 4800 --no_cpu_validation --reinit false --realloc false -b b10
```

Run all benchmarks
```
graalpython --jvm --polyglot benchmark_wrapper.py -d -i 30 
```

Run the CUDA version of all benchmarks
```
graalpython --jvm --polyglor benchmark_wrapper.py -d -i 30 -c
```

To print the Java Stack Trace in case of exceptions, add the following to Graalpython
```
graalpython --python.ExposeInternalSources --python.WithJavaStacktrace=1 --experimental-options
```

Profile a specific benchmark using `nvprof`. Running `nvprof` as `sudo` might not be required, see [here](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters).
 Note that the `graalpython` benchmark has the `--nvprof` flag, so that only the real computation is profiled (and not the benchmark initialization). 
 Additionally, provide `nvprof` with flags `--csv` to get a CSV output, and `--log-file bench-name_%p.csv"` to store the result.
  Not using the flag `--print-gpu-trace` will print aggregated results. Additional metrics can be collected by `nvprof` with e.g. `--metrics "achieved_occupancy,sm_efficiency"` ([full list](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#metrics-reference)). GPUs with architecture starting from Turing (e.g. GTX 1660 Super) no longer allow collecting metrics with `nvprof`, but `ncu` ([link](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)) and Nsight Compute ([link](https://developer.nvidia.com/nsight-compute)).
```
sudo /usr/local/cuda/bin/nvprof --profile-from-start off --print-gpu-trace --profile-child-processes  /path/to/graalpython --jvm --polyglot --grcuda.InputPrefetch --grcuda.ForceStreamAttach --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.ExecutionPolicy=default --grcuda.DependencyPolicy=with-const --grcuda.RetrieveParentStreamPolicy=disjoint benchmark_main.py -d -i 10 -n 4800 --no_cpu_validation --reinit false --realloc false -b b10d --block_size_1d 256 --block_size_2d 16 --nvprof
```

* Benchmarks are defined in the `projects/resources/python/benchmark/bench` folder, 
and you can create more benchmarks by inheriting from the `Benchmark` class. Single benchmarks are executed from `benchmark_main.py`, while running all benchmark is done through `benchmark_wrapper.py`
* The output of benchmarks is stored in a JSON (by default, located in `data/results`)
* The benchmarking suite, through `benchmark_main.py`, supports the following options
    1. `-d`, `--debug`: print to the console the results and details of each benchmark. False by default
    2. `-t`, `--num_iter`: number of times that each benchmark is executed, for each combination of options. 30 by default
    3. `-o`, `--output_path`: full path to the file where results are stored. By default results are stored in `data/results`,
    and the file name is generated automatically
    4. `--realloc`: if true, allocate new memory and rebuild the GPU code at each iteration. False by default
    5. `--reinit`: if true, re-initialize the values used in each benchmark at each iteration. True by default
    6. `-c`, `--cpu_validation`: if present, validate the result of each benchmark using the CPU (use `--no_cpu_validation` to skip it instead)
    7. `-b`, `--benchmark`: run the benchmark only for the specified kernel. Otherwise run all benchmarks specified in `benchmark_main.py`
    8. `-n`, `--size`: specify the input size of data used as input for each benchmark. Otherwise use the sizes specified in `benchmark_main.py`
    9. `-r`, `--random`: initialize benchmarks randomly whenever possible. True by default
	10. `--block_size_1d`: number of threads per block when using 1D kernels
	11. `--block_size_2d`: number of threads per block when using 2D kernels
	12. `-g`, `--number_of_blocks`: number of blocks in the computation
	13. `-p`, `--time_phases`: measure the execution time of each phase of the benchmark; note that this introduces overheads, and might influence the total execution time. Results for each phase are meaningful only for synchronous execution
	14. `--nvprof`: if present, enable profiling when using nvprof. For this option to have effect, run graalpython using nvprof, with flag '--profile-from-start off'
	

## DAG Scheduling Settings
The automatic DAG scheduling of GrCUDA supports different settings that can be used for debugging or to simplify the dependency computation in some circumstances

* `ExecutionPolicy`: this regulates the global scheduling policy;
 `default` uses the DAG for asynchronous parallel execution, while `sync` executes each computation synchronously and can be used for debugging or to measure the execution time of each kernel
* `DependencyPolicy`: choose how data dependencies between GrCUDA computations are computed;
`with_const` considers read-only parameter, while `default` assumes that all arguments can be modified in a computation
* `RetrieveNewStreamPolicy`: choose how streams for new GrCUDA computations are created;
 `fifo` (the default) reuses free streams whenever possible, while `always_new` creates new streams every time a computation should use a stream different from its parent
* `RetrieveParentStreamPolicy`: choose how streams for new GrCUDA computations are obtained from parent computations;
`default` simply reuse the stream of one of the parent computations, while `disjoint` allows parallel scheduling of multiple child computations as long as their arguments are disjoint
* `--grcuda.InputPrefetch`: if present, prefetch the data on GPUs with architecture starting from Pascal. In most cases, it improves performance.
* `--grcuda.ForceStreamAttach`: if present, force association between arrays and CUDA streams. True by default on architectures older than Pascal, to allow concurrent CPU/GPU computation. On architectures starting from Pascal, it can improve performance.
