# GrCUDA

GrCUDA exposes existing GPU kernels and host functions that accept device objects as parameters to all GraalVM languages through the Truffle Interop Library.
GrCUDA represents device objects as flat arrays of primitive types and makes them accessible as array-like TruffleObjects to other Truffle languages.

GrCUDA itself is an expression-oriented language. Most expressions in GrCUDA, however, simply evaluate to function objects that can then be used in the host languages.

Contents:

- [Device Arrays](#device-arrays)
- [Function Invocations](#function-invocations)
- [Callables](#callables)
- [Function Namespaces](#function-namespaces)
- [Kernel and Host Function Signatures](#kernel-and-host-function-signatures)
- [Built-in Functions](#built-in-functions)

## Device Arrays

Device arrays are flat multi-dimensional arrays of primitive types. 
The arrays are can be accessed from GPU kernels and native host functions that accept GPU pointers as parameter. 
The device arrays can also be accessed from host languages through the array constructs available in these host languages. 
The memory for the device array is CUDA-managed memory. 
Device arrays can be allocated through *array allocation expressions* and through the built-in `DeviceArray`
constructor function (see [built-in functions](#built-in-functions)).

### Lifetime Considerations for the current implementation

Device arrays are tied to garbage collector of the VM. 
Their underlying memory is allocated off-heap through the CUDA Runtime. 
Only a stub object containing the pointer to the off-heap memory, type, and size information etc, is kept on-heap.
Therefore, the heap utilization does not reflect the off-heap utilization.
A large device array does not have a large on-heap footprint. 
The garbage collection pass may not be initiated even though the memory utilization (off-heap) is high.

Just as for other native bindings, GrCUDA is not able to prevent GPU kernels or host functions from capturing pointers to device array objects or their elements. 
Because device arrays are managed by the garbage collector,
capturing of references in native code can potentially lead to dangling references.

The CUDA-managed memory of the device array can be **freed explicitly** by calling the `free()` method of `DeviceArray`. 
This will release the allocated memory through `cudaFree()`.
Once the underlying memory is freed, the `DeviceArray` enters a defunct state. 
All subsequent accesses with throw an exception. 
The Boolean property `isMemoryFreed` of `DeviceArray` can be checked whether the device array's memory buffer has already be freed.

### Array Allocation Expressions

Device arrays are allocated using a syntax that is similar to arrays C/C++. 
Multi-dimensional arrays use row-major (C style) layout. 
Use `DeviceArray(type, size, 'F')` to create arrays with column-major (Fortran style) order.

The array sizes must be compile-time constants, i.e., either an integer literal or a constant expression. 
One way to define the array size in array allocation expressions from host languages is through string interpolation (e.g., template strings in JavaScript).

```C
double[100]          // array of 10 double-precision elements
int[10 * 100]        // array of 1,000 32-bit integer elements
float[1000][28][28]  // array of 1000 x 28 x 28 float elements
```

The supported data types are:

| GrCUDA Type  | Truffle (Java) Type | Compatible C++ Types
|--------------|---------------------|----------------------
| `boolean`    | `boolean`           | `bool`, `unsigned char`
| `char`       | `byte`              | `char`, `signed char`
| `short`      | `short`             | `short`
| `int`        | `int`               | `int`
| `long`       | `long`              | `long`, `long long`
| `float`      | `float`             | `float`
| `double`     | `double`            | `double`

The polyglot expression returns a `DeviceArray` object for one-dimensional arrays or a `MultDimDeviceArray` for a multi-dimensional array to the host.

**Example in Python:**

```python
device_array = polyglot.eval(language='grcuda', string='double[5]')
device_array[2] = 1.0
matrix = polyglot.eval(language='grcuda', string='float[28][28]')
matrix[4][3] = 42.0
```

**Example in R:**

Note that arrays are 1-indexed in R.

```R
device_array <- eval.polyglot('grcuda', 'double[5]')
device_array[2:3] <- c(1.0, 2.0)
matrix <- eval.polyglot'grcuda', 'float[28][28]')
matrix[5, 4] <- 42.0
```

**Example in NodeJS/JavaScript:**

```JavaScript
device_array = Polyglot.eval('grcuda', 'double[5]')
device_array[2] = 1.0
matrix = Polyglot.eval('grcuda', 'float[28][28]')
matrix[4][3] = 42.0
```

**Example in Ruby:**

```ruby
device_array = Polyglot.eval('grcuda', 'double[5]')
device_array[2] = 1.0
matrix = Polyglot.eval('grcuda', 'float[28]28]')
matrix[4][3] = 42.0
```

For Java and the C-binding, element access is exposed through a function call syntax.

**Example in Java:**

```Java
Value deviceArray = polyglot.eval("grcuda", "double[5]");
double oldValue = (Double) deviceArray.getArrayElement(2);
deviceArray.setArrayElement(1, 1.0);
Value matrix = polyglot.eval("grcuda", "float[28][28]");
matrix.getArrayElement(4).setArrayElement(3, 42.0);
```

## Function Invocations

Function invocations are evaluated inside GrCUDA, then the return values are passed back to the host language. 
The argument expressions that can be used in GrCUDA language strings are currently limited to literals and constant integer expressions. 
For general function invocation, first create a GrCUDA expression that returns a [callable](#callables) back to the host language and then invoke the callable from the host language.

**Example in JavaScript:**

```javascript
// cudaDeviceReset() does not return a value back to JavaScript
Polyglot.eval('grcuda','cudaDeviceReset()')

// DeviceArray("float", 1000) returns a device array containing 1,000 floats
const deviceArray = Polyglot.eval('grcuda', 'DeviceArray("float", 1000)')
```

## Callables

Callables are Truffle objects that can be invoked from the host language.
A common usage pattern is to submit polyglot expression that return callables.
Identifiers are inside a namespace. 
CUDA Functions reside in the root namespace (see [built-in functions](#built-in-functions)).

For device arrays and GPU pointers that are passed as arguments to callables, GrCUDA automatically passes the underlying pointers to the native host or kernel functions.

**Example in Python:**

```Python
cudaDeviceReset = polyglot.eval(language='grcuda', string='cudaDeviceReset')
cudaDeviceReset()

cudaMalloccudaFree = polyglot.eval(language='grcuda', string='cudaFree')
mem = polyglot.eval(language='grcuda', string='cudaMalloc(100)')
cudaFree(mem)  # free the memory

createDeviceArray = polyglot.eval(language='grcuda', String='DeviceArray')
in_ints = createDeviceArray('int', 100)
out_ints = createDeviceArray('int', 100)

bind = polyglot.eval(language='grcuda', string='bind')
# bind to symbol of native host function, returns callable
inc = bind('<path-to-library>/libinc.so',
    'increment(in_arr: in pointer sint32 , out_arr: out pointer sint32, n: uint64): sint32')
inc(in_ints, out_ints, 100)

bindkernel = polyglot.eval(language='grcuda', string='bindkernel')
# bind to symbol of kernel function from binary, returns callable
inc_kernel = bindkernel("inc_kernel.cubin",
    'inc_kernel(in_arr: in pointer sint32 , out_arr: out pointer sint32, n: uint64)')
inc_kernel(160, 128)(out_ints, in_ints, 100)
```

## Function Namespaces

GrCUDA organizes functions in a hierarchical namespace, i.e. namespaces can be nested.
The user can register additional kernel and host function into this namespace.
The registration is not persisted across instantiations of the GrCUDA language context.

The root namespace is called `CU`. 
It can be received through a polyglot expression.

**Example in Python:**

```python
import polyglot

cu = polyglot.eval(language='grcuda', string='CU')
```

## Kernel and Host Function Signatures

GrCUDA needs to know the signature of native kernel functions and native host functions to correctly pass arguments. 
The signature is expressed in *NIDL (Native Interface Definition Language)* syntax.
The NIDL specification is used in `bind()` for host functions and in `bindkernel()` for kernel functions.
Multiple host functions or multiple kernel function specifications can be combined into one single `.nidl` file, which can then be passed to `bindall()`.
This call registers the bindings to all listed functions in the GrCUDA namespace.

The NIDL signature contains the name of the function and the parameters with their types.
Host functions also have a return type.
Since GPU kernel functions always return `void`, the return type is omitted for kernel functions in the NIDL syntax.
The parameters are primitive types passed by-value or are pointers to primitive values.
The parameter names can be chosen arbitrarily.
The parameter names are used to improve error messages. 
They do not affect the execution.

A complete list of all supported NIDL types and their mapping to Truffle and C++ types can be found in the [NIDL type mapping document](typemapping.md).

Pointers are used to refer to device arrays. 
Device array objects passed as arguments to kernels or host functions automatically decay into pointers.
Pointer are typed and have a direction `<direction> pointer <element type>`.
Allowed keywords for pointer directions are `in`, `out`, and `inout`.

NIDL              | C++
------------------|-----
`in pointer T`    | `const T*`
`out pointer T`   | `T*`
`inout pointer T` | `T*`

With the `async` execution policy and the `with-const` DependencyPolicy, using the `const` and `in` signals that the input will not be modified by the kernel.
As such, the scheduler will optimize its execution, for example by overlapping it with other kernels that access the same data but do not modify them.
GrCUDA does not currently check that the arrays are actually unmodified! 
It is responsibility of the user to use `const/in` correctly.

**Example Host Function Signatures:**

```text
saxpy(n: sint32, alpha: float, xs: in pointer float, ys: inout pointer float): void
```

The signature declares a host function `saxpy` that takes a signed 32-bit int `n`, a
single precision floating point value `alpha`, a device memory pointer `xs` to
constant float, a device memory pointer `ys` to float. The function does not return a value and,
therefore, has the return type `void`. GrCUDA looks for a C-style function definition
in the shared library, i.e., it searches for the symbol `saxpy`.

A matching function would be defined in C++ code as:

```c++
extern "C"
void saxpy(int n, float alpha, const float* xs, float *ys) { ... }
```

If the function is implemented in C++ without C-style export, its symbol name is mangled.
In this case, prefix the function definition with the keyword `cxx` which will instruct
GrCUDA to search for a C++ mangled symbol name.

```text
cxx predict(samples: in pointer double, labels: out pointer sint32, num_samples: uint32, weights: in pointer float, num_weights: uint32): sint32
```

This function returns a signed 32-bit integer. Due to the `cxx` keyword, GrCUDA looks for the
symbol with the mangled name `_Z7predictPKdPijPKfj` in the shared library.

C++ namespaces are also supported. The namespaces can be specified (using `::`).

```c++
namespace aa {
  namespace bb {
    namespace cc {
      float foo(int n, const int* in_arr, int* out_arr) { ... }
    }
  }
}
```

The NIDL signature that matches to the resulting mangled symbol name
`_ZN2aa2bb2cc3fooEiPKiPi` of the function is:

```text
cxx aa:bb:cc::foo(n: sint32, in_arr: in pointer sint32, out_arr: out pointer sint32): float
```

**Example Kernel Signatures:**

```text
update_kernel(gradient: in pointer float, weights: inout pointer float, num_weights: uint32)
```

This signature declaration refers to a kernel that takes a pointer `gradient` to constant float,
a pointer `weights` to float, and a value `num_weights` as an unsigned 32-bit int.
Note that there return type specification is missing because CUDA kernel functions do not return
any value. GrCUDA looks for the function symbol `update_kernel` in the cubin or PTX file.
`nvcc` uses C++ name mangling by default. The `update_kernel` function would therefore have to be
defined as `extern "C"` in the CUDA source code.

GrCUDA can be instructed search for the C++ mangled name by adding `cxx` keyword.

```text
cxx predict_kernel(samples: in pointer double, labels: out pointer sint32, num_samples: uint32, weights: in pointer float, num_weights: uint32)
```

GrCUDA then searches for the symbol `_Z14predict_kernelPKdPijPKfj`.

### Syntax NIDL Specification Files

Multiple declaration of host and kernel functions can be specified in a NIDL file and
bound into GrCUDA namespace in one single step.

The functions are collected in binding groups within `{  }`. The following binding groups
exist:

Binding Group                    | Syntax
---------------------------------|-----------------------------
host functions in C++ namespace  | `hostfuncs aa:bb:cc { ... }`
C++ host functions w/o namespace | `hostfuncs { ... }`
C-style host functions           | `chostfuncs { ... }`
kernel functions in C++ namespace| `kernels aa:bb:cc { ... }`
kernel functions w/o namespace   | `kernels { ... }`
C-style kernel functions         | `ckernels { ... }`

For C++ host functions and kernels the `cxx` prefix in the function signature is not used,
since it is already defined in the binding group.

**Example Binding Host Function from NIDL File:**

```console
$ cat foo_host_bindings.nidl
// host functions with C++ symbols in namespace
hostfuncs aa::bb::cc {
  // _ZN2aa2bb2cc20incr_outofplace_hostEPKiPfi
  incr_outofplace_host(in_dev_arr: in pointer sint32, out_dev_arr: out pointer float, num_elements: sint32): sint32

  // _ZN2aa2bb2cc19gpu_incr_inplaceEPfi
  incr_inplace_host(dev_arr: inout pointer float, num_elements: sint32): void
}

// host function with C++ symbols without namespace
hostfuncs {
  // _Z21cxx_incr_inplace_hostPfi
  cxx_incr_inplace_host(dev_arr: inout pointer float, num_elements: sint32): void
}

// host function with C symbol (extern "C")
chostfuncs {
  // c_incr_inplace_host
  c_incr_inplace_host(dev_arr: inout pointer float, num_elements: sint32): void
}
```

The following example in JavaScript uses the `foo_host_bindings.nidl` file in `bindall()`.
The function are registered in to `myfoo` namespace which is directly under the
root namespace. The function executed is `aa::bb:cc::incr_inplace_host()` in the
shared library `libfoo.so`.

```javascript
const cu = Polyglot.eval('grcuda', 'CU')

// host all host functions from libfoo.so into 'myfoo' namespace.
cu.bindall('myfoo', 'libfoo.so', 'foo_host_bindings.nidl')
...
// invoke functions in myfoo namespace
cu.myfoo.incr_inplace_host(deviceArray, deviceArray.length)
cu.myfoo.cxx_incr_inplace_host(deviceArray, deviceArray.length)
cu.myfoo.c_incr_inplace_host(deviceArray, deviceArray.length)
```

**Example Binding Kernel Function from NIDL File:**

```console
$ cat foo_kernel_bindings.nidl
// kernels with C++ symbols in namespace
kernels aa::bb::cc {
  // _ZN2aa2bb2cc22incr_outofplace_kernelEPKiPfi
  incr_outofplace_kernel(in_arr: in pointer sint32, out_arr: out pointer float,
    num_elements: sint32)

  // _ZN2aa2bb2cc19incr_inplace_kernelEPfi
  incr_inplace_kernel(arr: inout pointer float, num_elements: sint32)
}

// kernel with C++ symbol without namespace
kernels {
  // _Z23cxx_incr_inplace_kernelPfi
  cxx_incr_inplace_kernel(arr: inout pointer float, num_elements: sint32)
}

// kernel with C symbol (extern "C")
ckernels {
  // c_incr_inplace
  c_incr_inplace_kernel(arr: inout pointer float, num_elements: sint32)
}
```

The following example in JavaScript uses the `foo_kernel_bindings.nidl` file in `bindall()`.
The kernel functions are registered in to `myfoo` namespace which is directly under the
root namespace. The function executed is `aa::bb:cc::incr_inplace_host()` in the
shared library.

```javascript
const cu = Polyglot.eval('grcuda', 'CU')

// host all host functions from libfoo.so into 'myfoo' namespace.
cu.bindall('myfoo', '.so', 'foo_host_bindings.nidl')
...
// invoke functions in myfoo namespace
cu.myfoo.incr_inplace_host(deviceArray, deviceArray.length)
cu.myfoo.cxx_incr_inplace_host(deviceArray, deviceArray.length)
cu.myfoo.c_incr_inplace_host(deviceArray, deviceArray.length)
```

## Built-in Functions

The built-in functions are located in the root namespace of GrCUDA.
The functions are accessible directly in polyglot expression or, alternatively,
through the `CU` root namespace object.

### bind() Function

The `bind()` function allows binding a symbol which corresponds to a
host function from a shared library (.so) to a callable.
`bind()` returns the callable back to the host language.

```text
bind(libraryPath, nidlSignature): callable
```

`libraryPath`: is the full path to the .so file

`nidlSignature`: is the function's signature in NIDL syntax.
   Add prefix `cxx` if the symbol for function has C++ mangled name in the shared library.

**Example:**

```text
bind("<path-to-library>/libinc.so",
     "increment(arr_out: out pointer float, arr_in: pointer float, n_elements: uint64): sint32")
bind("<path-to-library>/libinc.so", "c",
     "cxx inc(arr_out: out pointer float, arr_in: pointer float, n_elements: uint64): sint32")
```

A complete example is given in the [bindings tutorial](docs/bindings.md).

### bindall() Function

Multiple host and kernel functions can be grouped and imported into rCUDA in one single step.
The signatures of host and kernel functions are specified NIDL files using
[NIDL syntax](#kernel-and-host-function-signatures). All listed functions are registered in
the GrCUDA `targetNamespace`.

```text
bindall(targetNamespace, fileName, nidlFileName)
```

`targetNamespace`: GrCUDA namespace into which all functions listed in the NIDL file that
  are imported from `fileName` are registered.

`nidlFileName`: name of the NIDL that contains specification for the host or kernel
  function.

- Note that the NIDL file can only contain specifications for host functions or kernel
functions, but not both.

- Note that for functions whose symbol have C++ mangled names need to be enclosed in
  `hostfuncs` and `kernels` groups. The `cxx` prefix as used in `bind()` and `bindkernel()`
  cannot used inside NIDL files.

**Examples:**

```text
bindall("myfoo", "libfoo.so", "foo_host_bindings.nidl")
bindall("myfoo", "foo.cubin", "foo_kernel_bindings.nidl")
```

A complete example is given in the [bindings tutorial](docs/bindings.md).

### bindkernel() Function

Kernel functions are bound to callable objects in the host
lost language. Kernel functions can be imported from `cubin` binary files
or PTX files. A kernel callable is bound to a `__global__` symbol in
a `cubin` or PTX file using the `bindkernel(..)`
function.

```text
bindkernel(cubinFile, nidlSignature)
```

`cubinFile`: name of the cubin file that was produced by `nvcc`

`nidlSignature`: is the function's signature in NIDL syntax. Since kernels do not return
  a result value, the return type is omitted.
  Add prefix `cxx` if the symbol for function has C++ mangled name in the shared library.

**Example:**

```text
bindkernel("inc_kernel.cubin", "inc_kernel(arr_out: out pointer float, arr_in: pointer float, n_elements: uint64)
```

A complete example is given in the [bindings tutorial](docs/bindings.md).

### buildkernel() Function

A GPU kernel callable can be created at runtime from a host language
string that contains the CUDA C/C++ source code.

```text
buildkernel(cudaSourceString, nidlSignature)
```

`cudaSourceString`: name of the cubin file that was produced by `nvcc`

`nidlSignature`: is the function's signature in NIDL syntax. Since kernels do not return
  a result value, the return type is omitted.
  The prefix `cxx` cannot he specified in build kernel as the NVRTC, used to compile
  the kernel, correctly resolves the lowered name itself.

If the kernel in `cudaSourceString` has a template parameter, the template argument
in the instantiation can be specified between `< >` the kernel name. The template
argument is passed to NVRTC. Thus, type arguments must be C++ types rather than
NIDL types. In the following example the template argument is `int` rather than
`sint32`.

**Example:**

```javascript
const kernelSource = `
template <typename T>
__global__ void inc_kernel(T *out_arr, const T *in_arr, int num_elements) {
  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;
       idx += gridDim.x * blockDim.x) {
    out_arr[idx] = in_arr[idx] + (T{} + 1);
  }
}`
const signature =
  'inc_kernel<int>(out_arr: out pointer sint32, in_arr: in pointer sint32, num_elments: sint32)'
const incKernel = cu.buildkernel(kernelSource, signature)
```

A complete example is given in the [bindings tutorial](docs/bindings.md).

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

```text
Property Name:

asyncEngineCount
canFlushRemoteWrites
canMapHostMemory
canUseHostPointerForRegisteredMem
clockRate
computeCapabilityMajor
computeCapabilityMinor
computeMode
computePreemptionSupported
concurrentKernels
concurrentManagedAccess
cooperativeLaunch
cooperativeMultiDeviceLaunch
deviceName
directManagedMemAccessFromHost
eccEnabled
freeDeviceMemory
globalL1CacheSupported
globalMemoryBusWidth
gpuOverlap
hostNativeAtomicSupported
hostRegisterSupported
integrated
isMultiGpuBoard
kernelExecTimeout
l2CacheSize
localL1CacheSupported`
managedMemory
maxBlockDimX
maxBlockDimY
maxBlockDimZ
maxGridDimX
maxGridDimY
maxGridDimZ
maxPitch
maxRegistersPerBlock
maxRegistersPerMultiprocessor
maxSharedMemoryPerBlock
maxSharedMemoryPerBlockOptin
maxSharedMemoryPerMultiprocessor
maxSurface1DLayeredLayers
maxSurface1DWidth
maxSurface2DHeight
maxSurface2DLayeredHeight
maxSurface2DLayeredLayers
maxSurface2DLayeredWidth
maxSurface2DWidth
maxSurface3DDepth
maxSurface3DHeight
maxSurface3DWidth
maxSurfaceCubemapLayeredLayers
maxSurfaceCubemapLayeredWidth
maxSurfaceCubemapWidth
maxTexture1DLayeredLayers
maxTexture1DLayeredWidth
maxTexture1DLinearWidth
maxTexture1DMipmappedWidth
maxTexture1DWidth
maxTexture2DGatherHeight
maxTexture2DGatherWidth
maxTexture2DHeight
maxTexture2DLayeredHeight
maxTexture2DLayeredLayers
maxTexture2DLayeredWidth
maxTexture2DLinearHeight
maxTexture2DLinearPitch
maxTexture2DLinearWidth
maxTexture2DMipmappedHeight
maxTexture2DMipmappedWidth
maxTexture2DWidth
maxTexture3DDepth
maxTexture3DDepthAlt
maxTexture3DHeight
maxTexture3DHeightAlt
maxTexture3DWidth
maxTexture3DWidthAlt
maxTextureCubemapLayeredLayers
maxTextureCubemapLayeredWidth
maxTextureCubemapWidth
maxThreadsPerBlock
maxThreadsPerMultiProcessor
memoryClockRate
multiGpuBoardGroupID
multiProcessorCount
pageableMemoryAccess
pageableMemoryAccessUsesHostPageTables
pciBusId
pciDeviceId
pciDomainId
singleToDoublePrecisionPerfRatio
streamPrioritiesSupported
surfaceAlignment
tccDriver
textureAlignment
texturePitchAlignment
totalConstantMemory
totalDeviceMemory
unifiedAddressing
warpSize
```

### CUDA Runtime Functions

A subset of the functions of the
[CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
are exported into the root namespace of GrCUDA i.e., the `CU` object.

```python
opaquePointerToDeviceMemory = polyglot.eval(language='grcuda', string='cudaMalloc(1000))')
...
cu = polyglot.eval(langauge='grcuda', string='CU')
cu.cudaFree(opaquePointerToDeviceMemory)
```

CUDA functions that take an out pointer, e.g., `​cudaError_t cudaMalloc(void** devPtr, size_t size)` , are wrapped such that they
return the result through the return value. In this case, errors are signaled
by throwing exceptions of type `GrCUDAException`, e.g., `cudaMalloc(size: uint64): pointer`.

CUDA Function          | NIDL Signature
-----------------------|---------------
cudaDeviceGetAttribute | `cudaDeviceGetAttribute(attr: sint32, device: sint32): sint32`
cudaDeviceReset        | `cudaDeviceReset(): void`
cudaDeviceSynchronize  | `cudaDeviceSynchronize(): void`
cudaFree               | `cudaFree(ptr: inout pointer void): void`
cudaGetDevice          | `cudaGetDevice(): sint32`
cudaGetDeviceCount     | `cudaGetDeviceCount(): sint32`
cudaGetErrorString     | `cudaGetErrorString(error: sint32): string`
cudaMalloc             | `cudaMalloc(size: uint64): pointer`
cudaMallocManaged      | `cudaMallocManaged(size: uint64, flags: uint32): pointer`
cudaSetDevice          | `cudaSetDevice(device: sint32): void`
cudaMemcpy             | `cudaMemcpy(dst: out pointer void, src: in pointer void, count: uint64, kind: sint32):void`

## cuBLAS Functions

The following functions from cuBLAS are are preregistered in in the namespace
`BLAS`. The `cublasHandle_t` is implicitly provided.
The location of the cuBLAS library (`libcublas.so`) can be set via the
`--grcuda.CuBLASLibrary= location`. cuBLAS support can be disabled with the option
`--grcuda.CuBLASEnabled=false`.

```text
// BLAS-1 AXPY:  y[i] := alpha * x[i] + y[i]
cublasSaxpy_v2(n: sint32, alpha: in pointer float, x: in pointer float, incx: sint32,
               y: inout pointer float, incy: sint32): sint32
cublasDaxpy_v2(n: sint32, alpha: in pointer double, x: in pointer double, incx: sint32,
               y: inout pointer double, incy: sint32): sint32
cublasCaxpy_v2(n: sint32, alpha: in pointer float, x: in pointer float, incx: sint32,
               y: inout pointer float, incy: sint32): sint32
cublasZaxpy_v2(n: sint32, alpha: in pointer double, x: in pointer double, incx: sint32,
               y: inout pointer double, incy: sint32): sint32

// BLAS-2 GEMV:  y[i] := alpha * trans(A) + beta * y[i]
cublasSgemv_v2(trans: sint32, n: sint32, m: sint32, alpha: in pointer float,
               A: in pointer float, lda: sint32,
               x: in pointer float, incx: sint32,
               beta: in pointer float,
               y: inout pointer float, incy: sint32): sint32
cublasDgemv_v2(trans: sint32, n: sint32, m: sint32, alpha: in pointer double,
               A: in pointer double, lda: sint32,
               x: in pointer double, incx: sint32,
               beta: in pointer double,
               y: inout pointer double, incy: sint32): sint32
cublasCgemv_v2(trans: sint32, n: sint32, m: sint32, alpha: in pointer float,
               A: in pointer float, lda: sint32,
               x: in pointer float, incx: sint32,
               beta: in pointer float,
               y: inout pointer float, incy: sint32): sint32
cublasZgemv_v2(trans: sint32, n: sint32, m: sint32, alpha: in pointer double,
               A: in pointer double, lda: sint32,
               x: in pointer double, incx: sint32,
               beta: in pointer double,
               y: inout pointer double, incy: sint32): sint32

// BLAS-3 GEMM:  C := alpha * transa(A) * transb(B) + beta * C
cublasSgemm_v2(transa: sint32, transb: sint32, n: sint32, m: sint32, k: sint32,
               alpha: in pointer float,
               A: in pointer float, lda: sint32,
               B: in pointer float, ldb: sint32,
               beta: in pointer float,
               C: in pointer float, ldc: sint32): sint32
cublasDgemm_v2(transa: sint32, transb: sint32, n: sint32, m: sint32, k: sint32,
               alpha: in pointer double,
               A: in pointer double, lda: sint32,
               B: in pointer double, ldb: sint32,
               beta: in pointer double,
               C: in pointer double, ldc: sint32): sint32
cublasCgemm_v2(transa: sint32, transb: sint32, n: sint32, m: sint32, k: sint32,
               alpha: in pointer float,
               A: in pointer float, lda: sint32,
               B: in pointer float, ldb: sint32,
               beta: in pointer float,
               C: in pointer float, ldc: sint32): sint32
cublasZgemm_v2(transa: sint32, transb: sint32, n: sint32, m: sint32, k: sint32,
               alpha: in pointer double,
               A: in pointer double, lda: sint32,
               B: in pointer double, ldb: sint32,
               beta: in pointer double,
               C: in pointer double, ldc: sint32): sint32
```

The letter S, D, C, Z designate the data type:

`{X}` | Type
------|-----
`S`   | single precision (32 bit) real number
`D`   | double precision (64 bit) real number
`C`   | single precision (32 bit) complex number
`Z`   | double precision (64 bit) complex number

## cuML Functions

Exposed functions from RAPIDS cuML (`libcuml.so`) are preregistered with
in the namespace `ML`. The `cumlHandle_t` argument, is implicitly provided by
GrCUDA and, thus, must be omitted in the polyglot callable.

The cuML function registry can be disabled by setting `--grcuda.CuMLEnabled=false`.
The absolute path to the `libcuml.so` shared library must be specified in  `--grcuda.CuMLLibrary=`.

Current set of **preregistered** cuML functions:

- `void ML::cumlSpDbscanFit(DeviceArray input, int num_rows, int num_cols, float eps, int min_samples, DeviceArray labels, size_t max_bytes_per_chunk, int verbose)`
- `void ML::cumlDbDbscanFit(DeviceArray input, int num_rows, int num_cols, double eps, int min_samples, DeviceArray labels, size_t max_bytes_per_chunk, int verbose)`

## Grammar of GrCUDA

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
