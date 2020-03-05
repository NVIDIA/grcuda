# grCUDA Language

grCUDA currently supports three different constructs:

- [Array Allocation Expressions](#array-expressions)
- [Function Invocations](#function-invocations)
- [Expressions that produce Callables](#callables) (for native function calls and GPU kernel launches)

## Array Allocation Expressions

Array allocation expressions define and allocate arrays that
accessible from both host and device code. The syntax is similar to arrays C/C++.

```C
double[100]          // array of 10 double-precision elements
int[10 * 100]        // array of 1,000 32-bit integer elements
float[1000][28][28]  // array of 1000 x 28 x 28 float elements
```

The supported data types are:

| grCUDA       | Java Type |
|--------------|-----------|
| `char`       | `byte`    |
| `short`      | `short`   |
| `int`        | `int`     |
| `long`       | `long`    |
| `float`      | `float`   |
| `double`     | `double`  |

The polyglot expression returns a `DeviceArray` object for
one-dimensional arrays or a `MultDimDeviceArray` for a multi-dimensional array to the host. The polyglot code can access
the array a like an ordinary array in the host language.
Device arrays allocated through device expressions are stored
in _row-major_ layout (C order). The `DeviceArray(..., 'F')` function,
described below, can be used to allocate multi-dimensional arrays
in _column-major_ (Fortran order).

### Example in Python

```python
device_array = polyglot.eval(language='grcuda', string='double[5]')
device_array[2] = 1.0
matrix = polyglot.eval(language='grcuda', string='float[28][28]')
matrix[4][3] = 42.0
```

### Example in R

Note that arrays are 1-indexed in R.

```R
device_array <- eval.polyglot('grcuda', 'double[5]')
device_array[2:3] <- c(1.0, 2.0)
matrix <- eval.polyglot'grcuda', 'float[28][28]')
matrix[5, 4] <- 42.0
```

### Example in Node/JavaScript

```JavaScript
device_array = Polyglot.eval('grcuda', 'double[5]')
device_array[2] = 1.0
matrix = Polyglot.eval('grcuda', 'float[28][28]')
matrix[4][3] = 42.0
```

### Example in Ruby

```ruby
device_array = Polyglot.eval('grcuda', 'double[5]')
device_array[2] = 1.0
matrix = Polyglot.eval('grcuda', 'float[28]28]')
matrix[4][3] = 42.0
```

For Java and the C-binding, element access is exposed to a function call syntax:

### Example in Java

```Java
Value deviceArray = polyglot.eval("grcuda", "double[5]");
double oldValue = (Double) deviceArray.getArrayElement(2);
deviceArray.setArrayElement(1, 1.0);
Value matrix = polyglot.eval("grcuda", "float[28][28]");
matrix.getArrayElement(4).setArrayElement(3, 42.0);
```

### Example in C

```C
void * device_array = polyglot_eval("grcuda", "double[5]");
double old_value = polyglot_as_double(
    polyglot_get_array_element(device_array, 1)));
polyglot_set_array_element(device_array, 1, 10);
void * matrix = polyglot_eval("grcuda", "float[28][28]");
polyglot_set_array_element(
  polyglot_get_array_element(matrix, 4),
  3, 42.0);
```

## Function Invocations

Function invocations are evaluated inside grCUDA the return
values are passed back to the host language. The argument expressions
that can be used in grCUDA language strings are currently limited to literals and constant integer expressions.
For general function invocation, first create a grCUDA expression that returns a [callable](#callables) back to the host language and then invoke the callable from the host language.

## Predefined grCUDA Functions

### bind() Function

The `bind()` function allows binding a symbol which corresponds to a
host function from a shared library (.so) to a callable.
`bind()` returns the callable back to the host language.

Signature:

```text
bind(libraryPath, symbolName, nfiSignature): callable
````

`libraryPath`: is the full path to the .so file

`symbolName`: is name of the function symbol, possibly C++ mangled

`nfiSignature`: is the function's signature in NFI format.

**Example:**

```test
bind("<path-to-library>/libinc.so", "increment", "(pointer, pointer, uint64): sint32")
```

**Example CUDA/C++ Code:**

```cpp
#include <cuda_runtime.h>
// GPU Kernel
template <typename T>
__global__ void inc_kernel(T *arr_out, const T *arr_in, size_t n_elements ) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements;
       i += blockDim.x * gridDim.x) {
    arr_out[i] = arr_in[i] + 1;
  }
}

extern "C" {
  cudaError_t increment(float *d_arr_out, float *d_arr_in, size_t n_elements) {
    // determine optimal block size
    int gridsize, blocksize;
    cudaError err = cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize,
    inc_kernel<float>);
    if (err != cudaSuccess)
      return err;
    int problem_gridsize = (n_elements + blocksize - 1) / blocksize;
    gridsize = (problem_gridsize < gridsize) ? problem_gridsize : gridsize;

    // invoke kernel
    inc_kernel<<<gridsize, blocksize>>>(d_arr_out, d_arr_in, n_elements);
    return cudaDeviceSynchronize();
  }
}
```

Build shared library:

```bash
 nvcc --shared --compiler-options -fPIC -o libinc.so incrementer.cu
```

**Example Invocation in Python:**

Creates callable for `int inc(float*, float*, size_t)` defined in in `libinc.so`.
The pointer arguments are automatically unboxed from the supplied device array objects.

```python
import polyglot
d_in = polyglot.eval(language='grcuda', string='float[100]')
for i in range(100):
    d_in[i] = i # numbers 0.0, 1.0, ..., 99.0
d_out = polyglot.eval(language='grcuda', string='float[100]')

# bind symbol, return callable
inc = polyglot.eval(
    language='grcuda',
    string="""bind("<path-to-library>/libinc.so",
        "increment", "(pointer, pointer, uint64): sint32")""")

# invoke function with device arrays
inc(d_out, d_in, 100)

print(d_out)    # numbers 1.0, 2.0, ..., 100.0
```

### CUDA Runtime Functions

A number of CUDA functions are accessible in the default namespace:

**Example in Python:**

```Python
      polyglot.eval(language='grcuda', string='cudaDeviceReset()')
      polyglot.eval(language='grcuda', string='cudaDeviceSynchronize()')
n   = polyglot.eval(language='grcuda', string='cudaGetDeviceCount()')
s   = polyglot.eval(language='grcuda', string='cudaGetErrorString(23)')
ptr = polyglot.eval(language='grcuda', string='cudaMalloc(10000)')
```

### bindkernel() Function

Kernel functions are bound to callable objects in the host
lost language. Kernel functions can be imported from `cubin` binary
files or generated from a CUDA C/C++ string that is compiled at
runtime. A kernel callable is bound to a `__global__` symbol in
a `cubin` file using the `bindkernel(..)`
function. The signature of the `bindkernel(..)` function is:

Signature:

```text
bindkernel(cubinFile, symbolName, signatureString)
```

`cubinFile`: name of the cubin file that was produced by `nvcc`

`symbolName`: is name of the function symbol

`signatureString`: is the signature the kernel function in grCUDA NFI format.

**Example:**

```text
bindkernel("inc_kernel.cubin", "inc_kernel", "pointer, pointer, uint64")
```

See description in the [polyglot kernel launch](docs/launchkernel.md) documentation for details.

### buildkernel() Function

A GPU kernel callable can be created at runtime from a host language
string that contains the CUDA C/C++ source code.

Signature:

```text
buildkernel(cudaSourceString, kernelName, signatureString)
```

`cudaSourceString`: name of the cubin file that was produced by `nvcc`

`kernelName`: is name of the kernel function

`signatureString`: is the signature of the kernel function in grCUDA NFI format.

**Example:**

```C
buildkernel(
  "__global__ void my_kernel(int i) { printf(\"%d\\n\", i); }",
  "my_kernel", "sint32")
```

See description in the [polyglot kernel launch](docs/launchkernel.md) documentation for details.

### getdevices() and getdevice() Functions

The `getdevices()` functions returns an array that contains all visible
CUDA devices. `getdevice(k)` returns the `k` visible device, with
`k` ranging from 0 to the number of visible devices - 1.

```text
devices = getdevices()
device = getdevice(deviceOrdinal)
```

`deviceOrdinal`: integer `k` that for the kth device, `k` from 0 to
the number of visible devices
(see [cudaGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html))

Both functions return `Devices` objects which have the following members:

Attribute `id`: the device ID (ordinal)

Attribute `properties`: property objects containing device attributes
returned by the CUDA runtime `cudaDeviceGetAttributeGet()`,
`cudaMemgetInfo()` and `cuDeviceGetName()`.

Method `isCurrent()`: method returns true iff `id` is the device
on which the currently active host thread executes device code.

Method `setCurrent()`: method sets `id` as the device the
currently active host thread should execute device code.

**Example:**

```Python
devices = polyglot.eval(language='grcuda', 'getdevices()')
device0 = polyglot.eval(language='grcuda', 'getdevice(0)')
# identical to device0 = devices[0]

for device in devices:
    print('{}: {}, {} multiprocessors'.format(device.id,
       device.property.deviceName,
       device.property.multiProcessorCount))
# example output
# 0: TITAN V, 80 multiprocessors
# 1: Quadro GP100, 56 multiprocessors
device0.setCurrent()
print(device0.isCurrent())  # true
```

Table: Device Properties Names (see also
[CUDA Runtime Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html))
| Property Name
|-------------------------------------------|
| `asyncEngineCount`                        |
| `canFlushRemoteWrites`                    |
| `canMapHostMemory`                        |
| `canUseHostPointerForRegisteredMem`       |
| `clockRate`                               |
| `computeCapabilityMajor`                  |
| `computeCapabilityMinor`                  |
| `computeMode`                             |
| `computePreemptionSupported`              |
| `concurrentKernels`                       |
| `concurrentManagedAccess`                 |
| `cooperativeLaunch`                       |
| `cooperativeMultiDeviceLaunch`            |
| `deviceName`                              |
| `directManagedMemAccessFromHost`          |
| `eccEnabled`                              |
| `freeDeviceMemory`                        |
| `globalL1CacheSupported`                  |
| `globalMemoryBusWidth`                    |
| `gpuOverlap`                              |
| `hostNativeAtomicSupported`               |
| `hostRegisterSupported`                   |
| `integrated`                              |
| `isMultiGpuBoard`                         |
| `kernelExecTimeout`                       |
| `l2CacheSize`                             |
| `localL1CacheSupported`                   |
| `managedMemory`                           |
| `maxBlockDimX`                            |
| `maxBlockDimY`                            |
| `maxBlockDimZ`                            |
| `maxGridDimX`                             |
| `maxGridDimY`                             |
| `maxGridDimZ`                             |
| `maxPitch`                                |
| `maxRegistersPerBlock`                    |
| `maxRegistersPerMultiprocessor`           |
| `maxSharedMemoryPerBlock`                 |
| `maxSharedMemoryPerBlockOptin`            |
| `maxSharedMemoryPerMultiprocessor`        |
| `maxSurface1DLayeredLayers`               |
| `maxSurface1DWidth`                       |
| `maxSurface2DHeight`                      |
| `maxSurface2DLayeredHeight`               |
| `maxSurface2DLayeredLayers`               |
| `maxSurface2DLayeredWidth`                |
| `maxSurface2DWidth`                       |
| `maxSurface3DDepth`                       |
| `maxSurface3DHeight`                      |
| `maxSurface3DWidth`                       |
| `maxSurfaceCubemapLayeredLayers`          |
| `maxSurfaceCubemapLayeredWidth`           |
| `maxSurfaceCubemapWidth`                  |
| `maxTexture1DLayeredLayers`               |
| `maxTexture1DLayeredWidth`                |
| `maxTexture1DLinearWidth`                 |
| `maxTexture1DMipmappedWidth`              |
| `maxTexture1DWidth`                       |
| `maxTexture2DGatherHeight`                |
| `maxTexture2DGatherWidth`                 |
| `maxTexture2DHeight`                      |
| `maxTexture2DLayeredHeight`               |
| `maxTexture2DLayeredLayers`               |
| `maxTexture2DLayeredWidth`                |
| `maxTexture2DLinearHeight`                |
| `maxTexture2DLinearPitch`                 |
| `maxTexture2DLinearWidth`                 |
| `maxTexture2DMipmappedHeight`             |
| `maxTexture2DMipmappedWidth`              |
| `maxTexture2DWidth`                       |
| `maxTexture3DDepth`                       |
| `maxTexture3DDepthAlt`                    |
| `maxTexture3DHeight`                      |
| `maxTexture3DHeightAlt`                   |
| `maxTexture3DWidth`                       |
| `maxTexture3DWidthAlt`                    |
| `maxTextureCubemapLayeredLayers`          |
| `maxTextureCubemapLayeredWidth`           |
| `maxTextureCubemapWidth`                  |
| `maxThreadsPerBlock`                      |
| `maxThreadsPerMultiProcessor`             |
| `memoryClockRate`                         |
| `multiGpuBoardGroupID`                    |
| `multiProcessorCount`                     |
| `pageableMemoryAccess`                    |
| `pageableMemoryAccessUsesHostPageTables`  |
| `pciBusId`                                |
| `pciDeviceId`                             |
| `pciDomainId`                             |
| `singleToDoublePrecisionPerfRatio`        |
| `streamPrioritiesSupported`               |
| `surfaceAlignment`                        |
| `tccDriver`                               |
| `textureAlignment`                        |
| `texturePitchAlignment`                   |
| `totalConstantMemory`                     |
| `totalDeviceMemory`                       |
| `unifiedAddressing`                       |
| `warpSize`                                |

### DeviceArray Constructor Function

In addition to arrays expression, device arrays can also be
constructed using the `DeviceArray` grCUDA function.
This function creates one- and multi-dimensional arrays.

```text
DeviceArray(element_type_str, num_elements_dim1, num_elements_dim2,
            ... num_elements_dimN);
```

Optionally, the storage order can be specified by an additional
string argument: `'C'` for row-major (C format) and `'F'` for
column-major (Fortran format).

```text
DeviceArray(element_type_str, num_elements_dim1, num_elements_dim2,
            ... num_elements_dimN, format_specifier);
```

**Example:**

```C
DeviceArray("float", 28, 28, "F")   // allocates 28 x 28 float array
                                    // in column-major (Fortran) order
```

## Callables

Callables are Truffle objects that can be invoked from the host language.
Expression that produce callables are currently limited to identifiers.
Identifiers are inside a name space. CUDA Functions reside in the default namespace.

For device arrays and GPU pointers that are passed as arguments
to callables, grCUDA automatically passes the underlying
pointers to the native host or kernel functions.

**Example in Python:**

```Python
cudaDeviceReset = polyglot.eval(language='grcuda', string='cudaDeviceReset')
cudaDeviceReset()

cudaFree = polyglot.eval(language='grcuda', string='cudaFree') # returns an opaque GPU pointer
mem = polyglot.eval(language='grcuda', string='cudaMalloc(100)')
cudaFree(mem)  # free the memory

DeviceArray = polyglot.eval(language='grcuda', String='DeviceArray')
in_ints = DeviceArray('int', 100)
out_ints = DeviceArray('int', 100)

# bind to symbol of native host function, returns callable
bind = polyglot.eval(language='grcuda', string='bind')
inc = bind('<path-to-library>/libinc.so',
        'increment', '(pointer, pointer, uint64): sint32')
inc(in_ints, out_ints, 100)

# bind to symbol of kernel function from binary, returns callable
bindkernel = polyglot.eval(language='grcuda', string='bindkernel')
inc_kernel = bindkernel("inc_kernel.cubin", 'inc_kernel',
        'pointer, pointer, sint32)')
inc_kernel(160, 128)(out_ints, in_ints, 100)
```

### cuML Function Registry

Exposed functions from RAPIDS cuML (`libcuml.so`) are preregistered with
the corresponding NFI signature in the namespace `ML`. The `cumlHandle_t` argument,
is implicitly provided by grCUDA and, thus, must be omitted in the polyglot callable.

The cuML function registry can be disabled by setting `--grcuda.CuMLEnabled=false`.
The absolute path to the `libcuml.so` shared library must be specified in  `--grcuda.CuMLLibrary=`.

Current set of **preregistered** cuML functions:

- `void ML::cumlSpDbscanFit(DeviceArray input, int num_rows, int num_cols, float eps, int min_samples, DeviceArray labels, size_t max_bytes_per_chunk, int verbose)`
- `void ML::cumlDbDbscanFit(DeviceArray input, int num_rows, int num_cols, double eps, int min_samples, DeviceArray labels, size_t max_bytes_per_chunk, int verbose)`

## Grammar of grCUDA

```text
expr ::= arrayExpression | funcCall | callable

arrayExpr    ::= TypeName ('[' constExpr ']')+
callExpr     ::= FunctionName '(' argList? ')'
callable     ::= ('::') Identifier
              |  Namespace '::' Identifier

argList      ::= argExpr (',' argExpr)*
argExpr      ::= Identifier | String | constExpr
constExpr    ::= constTerm ([+-] constTerm)*
constTerm    ::= constFactor ([*/%] constFactor)*
constFactor  ::= IntegerLiteral | '(' constExpr ')'

TypeName     ::= Identifier
FunctionName ::= Identifier
String       ::= '"' Character '"'
```
