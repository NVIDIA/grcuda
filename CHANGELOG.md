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
