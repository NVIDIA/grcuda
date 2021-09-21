# GrCUDA: Polyglot GPU Access in GraalVM

This Truffle language exposes GPUs to the polyglot [GraalVM](http://www.graalvm.org). The goal is to

1. Make data exchange between the host language and the GPU efficient without burdening the programmer.
2. Allow programmers to invoke _existing_ GPU kernels from their host language.

Supported and tested GraalVM languages:

- Python
- JavaScript/NodeJS
- Ruby
- R
- Java
- C and Rust through the Graal Sulong Component

A description of GrCUDA and its the features can be found in the [GrCUDA documentation](docs/grcuda.md).

The [bindings documentation](docs/bindings.md) contains a tutorial that shows
how to bind precompiled kernels to callables, compile and launch kernels.

**Additional Information:**

- [GrCUDA: A Polyglot Language Binding for CUDA in GraalVM](https://devblogs.nvidia.com/grcuda-a-polyglot-language-binding-for-cuda-in-graalvm/). NVIDIA Developer Blog,
  November 2019.
- [GrCUDA: A Polyglot Language Binding](https://youtu.be/_lI6ubnG9FY). Presentation at Oracle CodeOne 2019, September 2019.
- [Simplifying GPU Access](https://developer.nvidia.com/gtc/2020/video/s21269-vid). Presentation at NVIDIA GTC 2020, March 2020.
- [DAG-based Scheduling with Resource Sharing for Multi-task Applications in a Polyglot GPU Runtime](https://ieeexplore.ieee.org/abstract/document/9460491). Paper at IPDPS 2021 on the GrCUDA scheduler, May 2021. [Video](https://youtu.be/QkX0FHDRyxA) of the presentation.

## Using GrCUDA in the GraalVM

GrCUDA can be used in the binaries of the GraalVM languages (`lli`, `graalpython`, `js`, `R`, and `ruby`).
The JAR file containing GrCUDA must be appended to the classpath or copied into `jre/languages/grcuda` (Java 8) or `languages/grcuda` (Java 11) of the Graal installation. 
Note that `--jvm` and `--polyglot` must be specified in both cases as well.

The following example shows how to create a GPU kernel and two device arrays in JavaScript (NodeJS) and invoke the kernel:

```JavaScript
// build kernel from CUDA C/C++ source code
const kernelSource = `
__global__ void increment(int *arr, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    arr[idx] += 1;
  }
}`
const cu = Polyglot.eval('grcuda', 'CU') // get GrCUDA namespace object
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
nvcc --cubin --generate-code arch=compute_75,code=sm_75 kernel.cu
$GRAALVM_DIR/bin/graalpython --polyglot --jvm example.py
1
2
...
100
```

For more details on how to invoke existing GPU kernels, see the Documentation on [polyglot kernel launches](docs/launchkernel.md).

## Installation

GrCUDA can either be installed from an [existing release](#installation-from-an-existing-release), or built from the [source files](#installation-from-source-files) using the `mx` build tool.
In both cases, it is recommended to follow these [extra steps](#additional-installations-steps) to ensure that your installation is working properly.

### Installation from an existing release

The original version of GrCUDA can be downloaded as a binary JAR from [grcuda/releases](https://github.com/NVIDIA/grcuda/releases) and manually copied into a GraalVM installation.

1. Download GraalVM CE 21.2.0 for Linux `graalvm-ce-java11-linux-amd64-21.2.0.tar.gz`
   from [GitHub](https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-21.2.0/graalvm-ce-java11-linux-amd64-21.2.0.tar.gz) and untar it in your
   installation directory.

  ```console
  cd <your installation directory>
  wget https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-21.2.0/graalvm-ce-java11-linux-amd64-21.2.0.tar.gz
  tar xfz graalvm-ce-java11-linux-amd64-21.2.0.tar.gz
  rm graalvm-ce-java11-linux-amd64-21.2.0.tar.gz
  export GRAALVM_DIR=`pwd`/graalvm-ce-java11-21.2.0
  ```

2. Download the GrCUDA JAR from [grcuda/releases](https://github.com/NVIDIA/grcuda/releases). If using the official release, the latest features (e.g. the asynchronous scheduler) are not available. Instead, follow the guide below to install GrCUDA from the source code.

  ```console
  cd $GRAALVM_DIR/languages
  mkdir grcuda
  cp <download folder>/grcuda-0.1.0.jar grcuda
  ```

3. Test GrCUDA in Node.JS from GraalVM.

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

### Installation from source files

If you want to build GrCUDA yourself, instead of using an existing release, you will need a couple of extra steps. 
This section contains all the steps required to setup GrCUDA if your goal is to contribute to its development, or simply hack with it.
For simplicity, let's assume that your installation is done in your home directory, `~`.

If you are installing GrCUDA on a new machine, you can simply follow or execute `oci_setup/setup_from_scratch.sh`. Here we repeat the same steps, with additional comments. 
The installation process has been validated with CUDA 11.4 and Ubuntu 20.04.
The same `oci_setup` has a number of useful scripts to configure machines on OCI and easily use GrCUDA.

1. First, download GraalVM as above.

  ```console
  wget https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-21.2.0/graalvm-ce-java11-linux-amd64-21.2.0.tar.gz
  tar xfz graalvm-ce-java11-linux-amd64-21.2.0.tar.gz
  rm graalvm-ce-java11-linux-amd64-21.2.0.tar.gz
  export GRAALVM_DIR=~/graalvm-ce-java11-21.2.0
  ```

2. To build GrCUDA, you also need a custom JDK that is used to build GraalVM.

  ```console
  wget https://github.com/graalvm/labs-openjdk-11/releases/download/jvmci-21.2-b08/labsjdk-ce-11.0.12+6-jvmci-21.2-b08-linux-amd64.tar.gz
  tar xfz labsjdk-ce-11.0.12+6-jvmci-21.2-b08-linux-amd64.tar.gz
  rm labsjdk-ce-11.0.12+6-jvmci-21.2-b08-linux-amd64.tar.gz
  export JAVA_HOME=~/labsjdk-ce-11.0.12-jvmci-21.2-b08
  ```
  
3. GrCUDA requires the [mx build tool](https://github.com/graalvm/mx).
Clone the mx repository and add the directory into `$PATH`, such that the `mx` can be invoked from
the command line.
We checkout the commit corresponding to the current GraalVM release.

  ```console
  git clone https://github.com/graalvm/mx.git
  cd mx
  git checkout d6831ca0130e21b55b2675f7c931da7da10266cb
  cd ..
  ```

4. You might also want the source files for GraalVM CE, at the commit corresponding to the current release of GraalVM. This is not required for building, but if you want to modify GrCUDA's source code, it is useful to also have access to GraalVM's code.

  ```console
  git clone https://github.com/oracle/graal.git
  cd graal
  git checkout e9c54823b71cdca08e392f6b8b9a283c01c96571
  cd ..
  ```

5. Last but not least, build GrCUDA

  ```console
  cd <directory containing this README>
  mx build
  # Install it for Java 8+;
  mkdir -p $GRAAL_HOME/jre/languages/grcuda;
  cp mxbuild/dists/jdk1.8/grcuda.jar $GRAAL_HOME/languages/grcuda/.;
  ```

* `./install.sh` does this sequence of command for you, if you need to rebuild GrCUDA.

### Additional installations steps

1. **Setup your CUDA environment**
* Install CUDA and Nvidia drivers, for example following the steps [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)
* Add the following to your environment (assuming you have installed CUDA in the default `/usr/local` location, and using the `nvcc` compiler. Add these lines to `~/.bashrc` to make them permanent.

```console
export CUDA_DIR=/usr/local/cuda
export PATH=$PATH:$CUDA_DIR/bin
```

2. **Setup your GraalVM and GrCUDA environment**
* Add the following to your environment (assuming you have installed the releases mentioned in step 2 and 3). Add these lines to `~/.bashrc` to make them permanent.

```console
export PATH=~/mx:$PATH
export JAVA_HOME=~/labsjdk-ce-11.0.12-jvmci-21.2-b08
export GRAAL_HOME=~/graalvm-ce-java11-21.2.0
export GRAALVM_HOME=$GRAAL_HOME
export PATH=$GRAAL_HOME/bin:$PATH
export PATH=$JAVA_HOME/bin:$PATH
export GRCUDA_HOME=~/grcuda
```
* `source ~/.bashrc` to make changes available.

3. **Install languages for GraalVM** (optional, but recommended)

```console
gu available
gu install native-image
gu install llvm-toolchain
gu install python 
gu install nodejs
gu rebuild-images polyglot
```

* If Graalpython is installed, create a `virtualenv` for it

```console
graalpython -m venv ~/graalpython_venv
source ~/graalpython_venv/bin/activate
```

* Recommended: install `numpy` in Graalpython (required for running GrCUDA benchmarks)

```console
graalpython -m ginstall install setuptools;
graalpython -m ginstall install Cython;
graalpython -m ginstall install numpy;
```

4. **Run GrCUDA Unit tests** using 

```bash
mx unittest com.nvidia
# To run a specific test, you can use
mx unittest com.nvidia.grcuda.test.BuildKernelTest#testBuildKernelwithNFILegacytSignature
```

### Setup your IDE

To develop GrCUDA, you will greatly benefit from having an IDE that allows jumping between symbols and debugging individual tests. 
Here, we explain how to setup IntelliJ Idea.

  1. `mx ideinit` from `$GRCUDA_HOME`, to setup the IDE
  2. Open Idea and select *"open project"*, then open GrCUDA
  3. See this [guide](https://github.com/graalvm/mx/blob/master/docs/IDE.md) to configure the syntax checker
  4. In IntelliJ Idea, install the Python plugin with `Settings -> Plugin -> Search "Python"`, then do `Project Structure -> SDKs -> Create a new Python 3.8 Virtual Environment`, it is used by `mx`
  5. Select the right JVM. It should select automatically your `$JAVA_HOME`. Othewise, `Project Structures -> Modules -> Set the Module SDK (under Dependencies)` of `mx` and submodules to your Java SDK (e.g. `11`). You can pick either the `labsjdk` or `graalvm`.
      * This is also given by the `configure` option if you try to build the project in IntelliJ Idea before setting these options. Set your project Java SDK (e.g. `11`) for those missing modules
      * When building for the first time in Intellij Idea, you might get errors like `cannot use --export for target 1.8`, which means that some package is being build with Java 8.
      * For these packages, there are two possible solutions. Try either of them, and stick to the one that works for you

          a. For those packages (look at the log to find them), manually specify a more recent SDK (e.g. `11`) as you did in step above. If you get errors of missing symbols, follow IntelliJ's hints and export the requested packages

          b. Remove the exports. `File -> Settings -> Build ... -> Java Compiler`, then remove all the `--export` flags.
  7. To run tests:

      a. Go to `Run (top bar) -> Edit Configurations -> Edit configuration templates -> Junit`

      b. (Not always necessary) By default, Idea should use your `env`. If not, make sure to have the same. Update the `PATH` variable so that it can find `nvcc`, and export `$GRAAL_HOME`. See `setup_machine_from_scratch.sh` to find all the environment variables.

      c. Modify the template Junit test configuration adding `-Djava.library.path="$GRAAL_HOME/lib` (in Java 11) to the VM options to find `trufflenfi`

      d. In IntelliJ Idea, `Run -> Edit Configurations`. Create a new JUnit configuration set to `All in package` with `com.nvidia.grcuda` as module and `com.nvidia.grcuda.test` selected below. Add `-Djava.library.path="$GRAAL_HOME/lib"` (or your version of GraalVM) if it's not already in VM options. Specify the SDK by setting the GraalVM JRE in e.g. `$GRAAL_HOME`, if not specified already.   

## Execute performance tests using Graalpython

To measure the performance of GrCUDA on complex GPU applications, we have developed a custom benchmark suite, found in `projects/resources/python/benchmark`.
These are the same benchmarks used in the [DAG-based Scheduling with Resource Sharing for Multi-task Applications in a Polyglot GPU Runtime](https://ieeexplore.ieee.org/abstract/document/9460491) paper.

Run a single benchmark with custom settings
```console
graalpython --jvm --polyglot --experimental-options --grcuda.InputPrefetch --grcuda.ForceStreamAttach --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.ExecutionPolicy=async --grcuda.DependencyPolicy=with-const --grcuda.RetrieveParentStreamPolicy=disjoint benchmark_main.py -d -i 10 -n 4800 --no_cpu_validation --reinit false --realloc false -b b10
```

Run all benchmarks
```console
graalpython --jvm --polyglot benchmark_wrapper.py -d -i 30 
```

Run the CUDA version of all benchmarks
```console
graalpython --jvm --polyglot benchmark_wrapper.py -d -i 30 -c
```

To print the Java Stack Trace in case of exceptions, add the following to Graalpython
```console
graalpython --python.ExposeInternalSources --python.WithJavaStacktrace=1 --experimental-options <your-benchmark-command>
```

Profile a specific benchmark using `nvprof`. Running `nvprof` as `sudo` might not be required, see [here](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters).
The `graalpython` benchmark offers the `--nvprof` flag: if enable, only the real computation is profiled (and not the benchmark initialization). 
Additionally, provide `nvprof` with flags `--csv` to get a CSV output, and `--log-file bench-name_%p.csv"` to store the result.
Not using the flag `--print-gpu-trace` will print aggregated results.
Additional metrics can be collected by `nvprof` with e.g. `--metrics "achieved_occupancy,sm_efficiency"` ([full list](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#metrics-reference)).
GPUs with architecture starting from Turing (e.g. GTX 1660 Super) no longer allow collecting metrics with `nvprof`, but `ncu` ([link](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)) and Nsight Compute ([link](https://developer.nvidia.com/nsight-compute)).

```
sudo /usr/local/cuda/bin/nvprof --profile-from-start off --print-gpu-trace --profile-child-processes  /path/to/graalpython --jvm --polyglot --experimental-options --grcuda.InputPrefetch --grcuda.ForceStreamAttach --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.ExecutionPolicy=async --grcuda.DependencyPolicy=with-const --grcuda.RetrieveParentStreamPolicy=disjoint benchmark_main.py -d -i 10 -n 4800 --no_cpu_validation --reinit false --realloc false -b b10d --block_size_1d 256 --block_size_2d 16 --nvprof
```

* Benchmarks are defined in the `projects/resources/python/benchmark/bench` folder, 
and you can create more benchmarks by inheriting from the `Benchmark` class. 
Individual benchmarks are executed from `benchmark_main.py`, while running all benchmark is done through `benchmark_wrapper.py`
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
The automatic DAG scheduling of GrCUDA supports different settings that can be used for debugging or to simplify the dependency computation in some circumstances. These options are provided at startup, using `--experimental-options --grcuda.OptionName=value`.

* `ExecutionPolicy`: this regulates the global scheduling policy;
 `async` uses the DAG for asynchronous parallel execution, while `sync` executes each computation synchronously and can be used for debugging or to measure the execution time of each kernel
* `DependencyPolicy`: choose how data dependencies between GrCUDA computations are computed;
`with-const` considers read-only parameter, while `no-const` assumes that all arguments can be modified in a computation
* `RetrieveNewStreamPolicy`: choose how streams for new GrCUDA computations are created;
 `fifo` (the default) reuses free streams whenever possible, while `always-new` creates new streams every time a computation should use a stream different from its parent
* `RetrieveParentStreamPolicy`: choose how streams for new GrCUDA computations are obtained from parent computations;
`same-as-parent` simply reuse the stream of one of the parent computations, while `disjoint` allows parallel scheduling of multiple child computations as long as their arguments are disjoint
* `InputPrefetch`: if present, prefetch the data on GPUs with architecture starting from Pascal. In most cases, it improves performance.
* `ForceStreamAttach`: if present, force association between arrays and CUDA streams. True by default on architectures older than Pascal, to allow concurrent CPU/GPU computation. On architectures starting from Pascal, it can improve performance.

## Publications

If you use GrCUDA in your research, please cite the following publication(s). Thank you!

```
Parravicini, A., Delamare, A., Arnaboldi, M., & Santambrogio, M. D. (2021, May). DAG-based Scheduling with Resource Sharing for Multi-task Applications in a Polyglot GPU Runtime. In 2021 IEEE International Parallel and Distributed Processing Symposium (IPDPS) (pp. 111-120). IEEE.
```
