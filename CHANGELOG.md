# 2022-06-01

* Added scheduling DAG export functionality. It is now possible to retrieve a graphic version of the scheduling DAG of the execution by adding `ExportDAG` in the startup options. The graph will be exported in .dot format in the path specified by the user as option argument.
* This information can be leveraged to better understand the achieved runtime performance and to compare the schedules derived from different policies. Moreover, poorly written applications will results in DAGs with low-level of task-parallelism independently of the selected policy, suggesting designers to change their applications’ logic.


# 2022-04-15

* Updated install.sh to compute the interconnection graph
* Updated benchmark_wrapper to retrieve connection_graph correctly
* Enabled min-transfer-size test in benchmark_wrapper
* Benchmark_wrapper set for V100

# 2022-02-16

* Added logging for multiple computations (List of floats) on the same deviceID. This information could be used in future history-based adaptive scheduling policies.

# 2022-01-26

* Added mocked benchmarks: for each multi-gpu benchmark in our suite, there is a mocked version where we check that the GPU assignment is the one we expect. Added utility functions to easily test mocked benchmarks
simplified round-robin device selection policy, now it works more or less as before but it is faster to update when using a subset of devices
* Added threshold parameter for data-aware device selection policies. When using min-transfer-size or minmax/min-transfer-time, consider only devices that have at least 10% (or X %) of the requested data. Basically, if a device only has a very small amount of data already available it is not worth preferring it to other devices, and it can cause scheduling to converge to a unique device. See B9 and B11, for example.
* Updated python benchmark suite to use new options, and optimized initialization of B1 (it is faster now) and B9 (it didn't work on matrices with 50K rows, as python uses 32-bit array indexing)
* Fixed performance regression in DeviceArray access. For a simple python code that writes 160M values on a DeviceArray, performance went from 4sec to 20sec by using GraalVM 21.3 instead of 21.2. Reverted GraalVM to 21.2. Using non-static final Logger in GrCUDAComputationalElement increased time from 4sec to 130sec (not sure why, they are not created in repeated array accesses): fixed this regression.

# 2022-01-14

* Modified the "new stream creation policy FIFO" to simply reuse an existing free stream, without using a FIFO policy. Using FIFO did not give any benefit (besides a more predictable stream assignment), but it was more complex (we needed both a set and a FIFO, now we just use a set for the free streams)
* Added device manager to track devices. This is mostly an abstraction layer over CUDARuntime, and allows retrieving the currently active GPU, or retrieving a specific device.
  * DeviceManager is only a "getter", it cannot change the state of the system (e.g. it does not allow changing the current GPU)
  * Compared to the original multi-GPU branch, we have cleaner separation. StreamManager has access to StreamPolicy, StreamPolicy has access to DeviceManager. StreamManager still has access to the runtime (for event creation, sync etc.), but we might completely hide CUDARuntime inside DeviceManager to have even more separation.
* Re-added script to build connection graph. We might want to call it automatically from grcuda if the output CSV is not found. Otherwise we need to update the documentation to tell users how to use the script

# 2022-01-12

* Modified DeviceSelectionPolicy to select a device from a specified list of GPUs, instead of looking at all GPUs.
That's useful because when we want to reuse a parent's stream we have to choose among the devices used by the parents, instead of considering all devices.
* Added new SelectParentStreamPolicy where we find the parents' streams that can be reused, and then looks at the best device among the devices where these streams are, instead of considering all the devices in the system as in the previous policy. The old policy is still available.

# 2021-12-21, Release 2

* Added support for GraalVM 21.3.
* Removed `ProfilableElement` Boolean flag, as it was always true.

# 2021-12-09

* Replaced old isLastComputationArrayAccess" with new device tracking API
* The old isLastComputationArrayAccess was a performance optimization used to track if the last computation on an array was an access done by the CPU (the only existing CPU computations), to skip scheduling of further array accesses done by the CPU
* Implicitly, the API tracked if a certain array was up-to-date on the CPU or on the GPU (for a 1 GPU system).
* The new API that tracks locations of arrays completely covers the old API, making it redundant. If an array is up-to-date on the CPU, we can perform read/write without any ComputationalElement scheduling.
* Checking if an array is up-to-date on the CPU requires a hashset lookup. It might be optimized if necessary, using a tracking flag.

# 2021-12-06

* Fixed major bug that prevented CPU reads on read-only arrays in-use by the GPU. The problem appeared only on devices since Pascal.
* Started integrating API to track on which devices a certain array is currently up-to-date. Slightly modified from the original multi-GPU API.

# 2021-12-05

* Updated options in GrCUDA to support new multi-gpu flags.
* Improved initialization of ExecutionContext, now it takes GrCUDAOptionMap as parameter.
* Improved GrCUDAOptionMap testing, and integrated preliminary multi-GPU tests.
* Renamed GrCUDAExecutionContext to AsyncGrCUDAExecutionContext.
* Integrated multi-GPU features into CUDARuntime
* Improved interface to measure execution time of computationalelements (now the role of "ProfilableElement" is clearer, and execution time logging has been moved inside ComputationElement instead of using StreamManager)
* Improved manual selection of GPU
* Unsupported tests (e.g. tests for multiGPU if just 1 GPU is available) are properly skipped, instead of failing or completing successfully without info
temporary fix for GRCUDA-56: cuBLAS is disabled on pre-pascal if async scheduler is selected

# 2021-11-30

* Updated python benchmark suite to integrate multi-gpu code.
* Minor updates in naming conventions (e.g. using snake_case instead of CamelCase)
* We might still want to update the python suite (for example the output dict structure), but for now this should work.

# 2021-11-29

* Removed deprecation warning for Truffle's ArityException. 
* Updated benchmark suite with CUDAs multiGPU benchmarks. Also fixed GPU OOB in B9.

# 2021-11-21

* Enabled support for cuSPARSE
  * Added support for CSR and COO `spmv` and `gemvi`.
  * **Known limitation:** Tgemvi works only with single-precision floating-point arithmetics.

# 2021-11-17

* Added the support of precise timing of kernels, for debugging and complex scheduling policies
  * Associated a CUDA event to the start of the computation in order to get the elapsed time from start to the end
  * Added` ElapsedTime` function to compute the elapsed time between events, aka the total execution time
  * Logging of kernel timers is controlled by the `grcuda.TimeComputation` option, which is false by default
  * Implemented with the ProfilableElement class to store timing values in a hash table and support future business logic
* Updated documentation for the use of the new `TimeComputation` option in README
* Considerations:
  * `ProfilableElement` is profilable (`true`) by default, and any `ConfiguredKernel` is initialized with this configuration. To date, there isn't any use for a `ProfilableElement` that is not profilable (`false`)
  * To date, we are tracking only the last execution of a `ConfiguredKernel` on each device. It will be useful in the future to track all the executions and leverage this information in our scheduler
  
# 2021-11-15

* Added read-only polyglot map to retrieve grcuda options. Retrieve it with `getoptions`. Option names and values are provided as strings. Find the full list of options in `GrCUDAOptions`.

# 2021-11-04

* Enabled the usage of TruffleLoggers for logging the execution of grcuda code
    * GrCUDA is characterized by the presence of several different types of loggers, each one with its own functionality
    * Implemented GrCUDALogger class is in order to have access to loggers of interest when specific features are needed
* Changed all the print in the source code in log events, with different logging levels
* Added documentation about logging in docs

# 2021-10-13

* Enabled support for cuBLAS and cuML in the async scheduler
  * Streams' management is now supported both for CUML and CUBLAS
  * This feature can be possibly applied to any library, by extending the `LibrarySetStreamFunction` class
* Set TensorRT support to experimental
  * TensorRT is currently not supported on CUDA 11.4, making it impossible to use along a recent version of cuML
  * **Known limitation:** due to this incompatibility, TensorRT is currently not available on the async scheduler

# 2021-09-30, Release 1

## API Changes

* Added option to specify arguments in NFI kernel signatures as `const`
    * The effect is the same as marking them as `in` in the NIDL syntax
    * It is not strictly required to have the corresponding arguments in the CUDA kernel marked as `const`, although
      that's recommended
    * Marking arguments as `const` or `in` enables the async scheduler to overlap kernels that use the same read-only
      arguments

## New asynchronous scheduler

* Added a new asynchronous scheduler for GrCUDA, enable it with `--experimental-options --grcuda.ExecutionPolicy=async`
    * With this scheduler, GPU kernels are executed asynchronously. Once they are launched, the host execution resumes
      immediately
    * The computation is synchronized (i.e. the host thread is stalled and waits for the kernel to finish) only once GPU
      data are accessed by the host thread
    * Execution of multiple kernels (operating on different data, e.g. distinct DeviceArrays) is overlapped using
      different streams
    * Data transfer and execution (on different data, e.g. distinct DeviceArrays) is overlapped using different streams
    * The scheduler supports different options, see `README.md` for the full list
    * It is the scheduler presented in "DAG-based Scheduling with Resource Sharing for Multi-task Applications in a
      Polyglot GPU Runtime" (IPDPS 2021)

## New features

* Added generic AbstractArray data structure, which is extended by DeviceArray, MultiDimDeviceArray,
  MultiDimDeviceArrayView, and provides high-level array interfaces
* Added API for prefetching
    * If enabled (and using a GPU with architecture newer or equal than Pascal), it prefetches data to the GPU before
      executing a kernel, instead of relying on page-faults for data transfer. It can greatly improve performance
* Added API for stream attachment
    * Always enabled in GPUs with with architecture older than Pascal, and the async scheduler is active. With the sync
      scheduler, it can be manually enabled
    * It restricts the visibility of GPU data to the specified stream
    * In architectures newer or equal than Pascal it can provide a small performance benefit
* Added `copyTo/copyFrom` functions on generic arrays (Truffle interoperable objects that expose the array API)
    * Internally, the copy is implemented as a for loop, instead of using CUDA's `memcpy`
    * It is still faster than copying using loops in the host languages, in many cases, and especially if host code is
      not JIT-ted
    * It is also used for copying data to/from DeviceArrays with column-major layout, as `memcpy` cannot copy
      non-contiguous data

## Demos, benchmarks and code samples

* Added demo used at SeptembeRSE 2021 (`demos/image_pipeline_local` and `demos/image_pipeline_web`)
    * It shows an image processing pipeline that applies a retro look to images. We have a local version and a web
      version that displays results a in web page
* Added benchmark suite written in Graalpython, used in "DAG-based Scheduling with Resource Sharing for Multi-task
  Applications in a Polyglot GPU Runtime" (IPDPS 2021)
    * It is a collection of complex multi-kernel benchmarks meant to show the benefits of asynchronous scheduling.

## Miscellaneosus

* Added dependency to `grcuda-data` submodule, used to store data, results and plots used in publications and demos.
* Updated name "grCUDA" to "GrCUDA". It looks better, doesn't it?
* Added support for Java 11 along with Java 8
* Added option to specify the location of cuBLAS and cuML with environment variables (`LIBCUBLAS_DIR` and `LIBCUML_DIR`)
* Refactored package hierarchy to reflect changes to current GrCUDA (e.g. `gpu -> runtime`)
* Added basic support for TruffleLogger
* Removed a number of existing deprecation warnings
* Added around 800 unit tests, with support for extensive parametrized testing and GPU mocking
* Updated documentation
    * Bumped GraalVM version to 21.2
    * Added scripts to setup a new machine from scratch (e.g. on OCI), plus other OCI-specific utility scripts (
      see `oci_setup/`)
    * Added documentation to setup IntelliJ Idea for GrCUDA development
    * Added documentation about Python benchmark suite
    * Added documentation on asynchronous scheduler options
