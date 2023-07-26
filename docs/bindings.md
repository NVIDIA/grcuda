# Tutorial: Polyglot Bindings to Kernel and Host Functions

GPU kernels and host function can be executed as function calls.
The corresponding functions are callable objects that are bound
to the respective kernel or host functions.
GrCUDA provides different ways to define these bindings:

- `bind(shareLibraryFile, functionNameAndSignature)` returns a callable
  object to the specified host function defined in the shared library (.so file).
- `bindkernel(fileName, kernelNameAndSignature)` returns a callable object
  to specified kernel function defined in PTX or cubin file.
- `bindall(targetNamespace, fileName, nidlFileName)` registers all functions
  listed in the NIDL (Native Interface Definition Language) for the
  specified binary file into the target namespace of GrCUDA.

The first two approaches are useful to implement the one-off binding to
a native function, be it a kernel or a host function. `bindall()` is use
to bind multiple functions from the same binary or PTX file. This tutorial shows
how to call existing native host functions and kernels from GraalVM languages
through GrCUDA.

## Binding and Invoking prebuilt Host Functions

Host functions can be bound from existing shared libraries by `bind()` or
`bindall()`. The former returns one single native function as a callable object
whereas later binds can be used to bind multiple functions into a specified
namespace within GrCUDA.

This simple example shows how to call two host functions from a shared library.
One function is defined a C++ namespace. The other function is defined as
`extern "C"`, i.e,. it is exported via C ABI. The two functions take a
pointer to device as argument and just launch the same kernel.

The source code for the shared library is as follows.

```c++
__global__ void inc_kernel(int * inout_arr, int num_elements) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_elements;
       i += gridDim.x * blockDim.x) {
    inout_arr[i] += 1;
  }
}

static cudaError_t check_status_and_sync() {
  auto status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  return status;
}

namespace aa {
namespace bb {
  // function defined in C++ namespace
  int cxx_inc_host(int * dev_inout_arr, int num_elements, int num_blocks, int block_size) {
    inc_kernel<<<num_blocks, block_size>>>(dev_inout_arr, num_elements);
    return check_status_and_sync();
  }
} // end of namespace bb
} // end of namespace aa

//function defined with C linkage
extern "C"
int c_inc_host(int * dev_inout_arr, int num_elements, int num_blocks, int block_size) {
  inc_kernel<<<num_blocks, block_size>>>(dev_inout_arr, num_elements);
  return check_status_and_sync();
}
```

Build the shared library (Linux).

```bash
nvcc -o libincrement.so -std=c++11 --shared -Xcompiler -fPIC increment.cu
```

`bind()` can be used to "import" a single function into GrCUDA as shown in
the following NodeJS/JavaScript example.

```javascript
const cu = Polyglot.eval('grcuda', 'CU')

// bind to C++ function in namespace aa:bb
const cxx_inc_host = cu.bind('libincrement.so',
  `cxx aa::bb::cxx_inc_host(inout_arr: inout pointer sint32, n_elements: sint32,
                            n_blocks: sint32, block_size: sint32): sint32`)

// bind to extern "C" function
const c_inc_host = cu.bind('libincrement.so',
  `c_inc_host(inout_arr: inout pointer sint32, n_elements: sint32,
              n_blocks: sint32, block_size: sint32): sint32`)

const n = 100
let deviceArray = cu.DeviceArray('int', n)
for (let i = 0; i < n; i++) {
  deviceArray[i] = i
}

const res1 = cxx_inc_host(deviceArray, n, 32, 256)
const res2 = c_inc_host(deviceArray, n, 32, 256)
console.log(`res1: `, res1)
console.log(`res2: `, res2)

for (const el of deviceArray) {
  console.log(el)
}
```

`bind()` takes the name (or path) of the shared library. The second argument
specifies the signature in NIDL format. Add the keyword `cxx` for C++ style functions. The C++ namespace can be specified using `::`. Without `cxx`
GrCUDA assumes C linkage of the function and does not apply any name mangling.
`bind()` returns the function objects as callables, i.e., `TruffleObject`
instances for which `isExecutable()` is `true`.

The two functions can be bound in one single step using `bindall()` as shown below:

```javascript
...
// bind host functions into namespace 'increment'
cu.bindall('increment', 'libincrement.so', 'increment_host.nidl')

const res1 = cu.increment.cxx_inc_host(deviceArray, n, 32, 256)
const res2 = cu.increment.c_inc_host(deviceArray, n, 32, 256)
...
```

The bindings are specified in a file `increment_host.nidl`:

```text
hostfuncs aa::bb {
  cxx_inc_host(inout_arr: inout pointer sint32, n_elements: sint32,
               n_blocks: sint32, block_size: sint32): sint32
}

chostfuncs {
  c_inc_host(inout_arr: inout pointer sint32, n_elements: sint32,
             n_blocks: sint32, block_size: sint32): sint32
}
```

The C++ functions are enclosed in `hostfuncs` scope. The C++ namespace
(e.g., `aa::bb` in the example above) is specified with the scope.
Functions with C linkage have to be placed inside a `chostfunc` scope.

All functions will be registered with their name as it appears
in the NIDL under the namespace that is provided as the first argument of
`bindall()`, e.g., `increment`.

## Binding and Launching prebuilt GPU Kernels

Existing CUDA kernels that are complied into `cubin` binaries or
PTX assembly files can be bound to callable objects using the `bindkernel()`
function and subsequently invoked from the various host languages in
the Graal/Truffle ecosystem. This is analogous to `bind()` for host functions.

As in previous example for external host functions, we will also use two
kernels. One kernel is defined in a C++ namespace, the other is defined
with C linkage. The kernels are compiled into a cubin binary.

The source code for the cubin is as follows.

```c++
namespace aa {
namespace bb {
// kernel in C++ namespace
__global__ void inc_kernel(int * inout_arr, int num_elements) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_elements;
       i += gridDim.x * blockDim.x) {
    inout_arr[i] += 1;
  }
}

} // end of namespace bb
} // end of namespace aa

// kernel with C linkage
extern "C"
__global__ void c_inc_kernel(int * inout_arr, int num_elements) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_elements;
       i += gridDim.x * blockDim.x) {
    inout_arr[i] += 1;
  }
}
```

The kernels are compiled into a cubin binary by `nvcc`. Note that architecture
and code version in `gencode` must match your generation. The options below
are for a GeForce RTX device.

```bash
nvcc -cubin -gencode=arch=compute_75,code=sm_75 \
   -o increment.cubin increment_kernels.cu
```

`bindkernel()` "imports" a single kernel function into GrCUDA. `bindkernel()`
returns the kernel as a callable object. It can be called like a function.
The parameters are the kernel grid size and as optional the amount dynamic shared
memory. This is analogous to the kernel launch configuration in CUDA that is
specified between `<<< >>>`.

The following example show how the kernels are bound and called from NodeJS JavaScript.

```javascript
const cu = Polyglot.eval('grcuda', 'CU')

// bind to kernel function in namespace aa:bb
const inc_kernel = cu.bindkernel('increment.cubin',
  `cxx aa::bb::inc_kernel(inout_arr: inout pointer sint32, n_elements: sint32)`)

// bind to kernel defined with C linkage
const c_inc_kernel = cu.bindkernel('increment.cubin',
  `c_inc_kernel(inout_arr: inout pointer sint32, n_elements: sint32)`)

const n = 100
let deviceArray = cu.DeviceArray('int', n)
for (let i = 0; i < n; i++) {
  deviceArray[i] = i
}

const numBlocks = 32
const blockSize = 256

inc_kernel(numBlocks, blockSize)(deviceArray, n)
c_inc_kernel(numBlocks, blockSize)(deviceArray, n)

for (const el of deviceArray) {
  console.log(el)
}
```

In the host language, kernel launches are function evaluations with two argument
lists, the first specifying the kernel grid, the second the arguments that are
passed to the kernel.
`inc_kernel(numBlocks, blockSize)(deviceArray, n)` corresponds to
```inc_kernel<<<numBlocks, blockSize>>>(deviceArrayPointer, n)``` in CUDA C/C++.

Just as for host functions, multiple kernel functions can be bound at the same time
using `bindall()` from a `cubin` or PTX file.

```javascript
const cu = Polyglot.eval('grcuda', 'CU')

// bind kernel functions into namespace 'increment'
cu.bindall('increment', 'increment.cubin', 'increment_kernels.nidl')

const n = 100
let deviceArray = cu.DeviceArray('int', n)
for (let i = 0; i < n; i++) {
  deviceArray[i] = i
}

const numBlocks = 32
const blockSize = 256

cu.increment.inc_kernel(numBlocks, blockSize)(deviceArray, n)
cu.increment.c_inc_kernel(numBlocks, blockSize)(deviceArray, n)

for (const el of deviceArray) {
  console.log(el)
}
```

The bindings are specified in a file `increment_kernels.nidl`:

```text
kernels aa::bb {
  inc_kernel(inout_arr: inout pointer sint32, n_elements: sint32)
}

ckernels {
  c_inc_kernel(inout_arr: inout pointer sint32, n_elements: sint32)
}
```

If the a kernel function is not declared with the `extern "C"`
`nvcc` generates C++ symbols for kernel functions. Such kernels can be enclosed
in a `kernels` scope in the NIDL file and subsequently bound in one step.
As in `hostfuncs` for C++ host functions, a C++ namespace can also be
specified in `kernels`. GrCUDA the searches all functions within the scope
in this namespace.

Kernel function defined with `extern "C"` can bound in a `ckernels` scope.
This scope specifies that no name mangling is used for kernels within the
scope.

All kernel function will be registered with their name as it appears in the
NIDL under the namespace that is provided as the first argument of bindall(),
e.g., `increment`.

## Runtime-compilation of GPU Kernels from CUDA C/C++

GrCUDA can also compile GPU kernels directly from CUDA C/C++
source code passed as a host-string argument to
`buildkernel(..)`. The signature of the function is:

```text
bindkernel(cudaSourceString, signatureString)
```

The `cudaSourceString` argument contains the kernel source code
in CUDA C/C++ as a host-language string.

The second argument, `signatureString`, specifies signature of the
kernel function, i.e., the name of the function and its parameters.
The syntax of the signature string is identical as in the corresponding argument
in the `bindkernel(..)` function.

Here is an example in Python that defines kernel function `inc_kernel`
as a CUDA C/C++ string and uses `bindkernel()`, which in turn leverages
the NVIDIA Runtime Compiler (NVRTC) to create a callable kernel object.

```python
import polyglot

cu = polyglot.eval(language='grcuda', string='CU')

kernel_source = """
__global__
void inc_kernel(int* arr_inout, size_t n_elements) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements;
       i += blockDim.x * gridDim.x) {
    arr_inout[i] += 1;
  }
}"""
inc_kernel = cu.buildkernel(kernel_source,
    'inc_kernel(arr_inout: inout pointer sint32, n_elements: uint64)')

n = 100
device_array = cu.DeviceArray('int', n)
for i in range(n):
    device_array[i] = i

num_blocks = 32
block_size = 256
inc_kernel(num_blocks, block_size)(device_array, n)

print(device_array)
```

## Launching Kernels

Once a kernel function is bound to a callable host-object or registered as
a function within GrCUDA, it can be launched like a function with two argument lists (for exceptions in Ruby and Java and Ruby see the examples below).

```test
kernel(num_blocks, block_size)(arg1, ..., argN)
```

This form is based on the triple angle brackets syntax of CUDA C/C++.

```C
kernel<<<num_blocks, block_size>>>(arg1, ..., argN)
```

The first argument list corresponds to the launch configuration, i.e.,
the kernel grid (number of blocks) and the block sizes (number of
threads per block).

GrCUDA now also supports asynchronous kernel launches, thanks to a computation DAG that allows scheduling parallel computations on different streams and avoid synchronization when not necessary.

__Examples:__

```Python
num_blocks = 160
block_size = 1024
# concatentated invocation using both argument lists
kernel(num_blocks, block_size)(out_arr, in_arr, num_elements)

# kernel with launch config yields in a kernel with a configured grid
configured_kernel = kernel(num_blocks, block_size)
configured_kernel(out_arr, in_ar, num_elements)
```

GrCUDA also supports 2D and 3D kernel grids that are specified
with the `dim3` in CUDA C/C++. In GrCUDA `num_blocks` and `block_size`
can be integers for 1-dimensional kernels or host language sequences
of length 1, 2, or 3 (Lists or Tuples in Python, Arrays in JavaScript
and Ruby, and vectors in R)

```Python
matmult((num_blocks_x, num_blocks_y),
        (block_size_x, block_size_y))(matrix_A, matrix_B, n, m, k)
fft3d((nx, ny, nz), (bx, by, bz))(matrix_A, matrix_B, n, m, k)
```

Additionally, the number of bytes to allocate for shared memory
per thread block is passed as the third argument in the
kernel configuration call, as in CUDA's `<<< >>>` notation.

```Python
num_shmem_bytes = 1024
kernel_using_shmem(num_blocks, block_size, num_shmem_bytes)(
                   inputA, inputB, results, N)
```

The syntax of most Graal language allows a concatenated invocation
with both argument lists. The exceptions are Ruby and Java:

In Ruby, callables need to be invoked using the `call(..)` method,
resulting in the following chained invocation:

```Ruby
kernel.call(num_blocks, block_size).call(out_arr, in_arr, num_elements)
```

For Java, the `execute()` needs to be used to invoke a Truffle callable.
Thus, a kernel needs to be launched as follows from Java:

```Java
kernel.execute(numBlocks, blockSize).execute(outArr, inArr, numElements)
```

## Host Language Examples

This section shows how to bind an existing kernel function
and launching the kernel from different host languages.

### Python

```python
import polyglot

cu = polyglot.eval(language='grcuda', string='CU')

inc_kernel = cu.bindkernel('increment.cubin',
    'c_inc_kernel(arr_inout: inout pointer sint32, n_elements: uint64)')

n = 100
device_array = cu.DeviceArray('int', n)
for i in range(n):
    device_array[i] = i

num_blocks = 32
block_size = 256
inc_kernel(num_blocks, block_size)(device_array, n)

print(device_array)
```

### NodeJS/JavaScript

```JavaScript
const cu = Polyglot.eval('grcuda', 'CU')

const inc_kernel = cu.bindkernel('increment.cubin',
  'c_inc_kernel(inout_arr: inout pointer sint32, n_elements: sint32)')

const n = 100
let deviceArray = cu.DeviceArray('int', n)
for (let i = 0; i < n; i++) {
  deviceArray[i] = i
}

const numBlocks = 32
const blockSize = 256
inc_kernel(numBlocks, blockSize)(deviceArray, n)

for (const el of deviceArray) {
  console.log(el)
}
```

### Ruby

```ruby
cu = Polyglot.eval("grcuda", "CU")

inc_kernel = cu.bindkernel("increment.cubin",
  "c_inc_kernel(inout_arr: inout pointer sint32, n_elements: sint32)")

num_elements = 100
device_array = cu.DeviceArray("int", num_elements)
for i in 0...num_elements do
  device_array[i] = i
end

num_blocks = 32
block_size = 256
inc_kernel.call(num_blocks, block_size).call(device_array, num_elements)

puts (0...num_elements).reduce("") {
  |agg, i| agg + device_array[i].to_s + " "
}
```

### R

```R
cu <- eval.polyglot('grcuda', 'CU')

inc_kernel <- cu$bindkernel('increment.cubin',
  'c_inc_kernel(inout_arr: inout pointer sint32, n_elements: sint32)')

num_elements <- 100
device_array <- cu$DeviceArray('int', num_elements)
for (i in 1:num_elements) {
  device_array[i] = i
}

num_blocks <- 32
block_size <- 256
inc_kernel(num_blocks, block_size)(device_array, num_elements)

print(device_array)
```

### Java

```Java
import java.util.stream.IntStream;
import java.util.stream.Collectors;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

class Increment {
  public static void main(String[] args) {
    Context polyglot = Context.newBuilder().allowAllAccess(true).build();
    Value cu = polyglot.eval("grcuda", "CU");

    Value incKernel = cu.invokeMember("bindkernel", "increment.cubin",
        "c_inc_kernel(inout_arr: inout pointer sint32, n_elements: sint32)");

    int numElements = 1000;
    Value deviceArray = cu.invokeMember("DeviceArray", "int", numElements);
    for (int i = 0; i < numElements; ++i) {
      deviceArray.setArrayElement(i, i);
    }

    int gridSize = 32;
    int blockSize = 256;
    incKernel.execute(gridSize, blockSize).execute(deviceArray, numElements);

    System.out.println(
      IntStream.range(0, numElements)
        .mapToObj(deviceArray::getArrayElement)
        .map(Object::toString)
        .collect(Collectors.joining(", "))
    );
  }
}
```

### C through Sulong

```C
#include <stdio.h>
#include <polyglot.h>

int main() {
  void * cu = polyglot_eval("grcuda", "CU");

  const char * charset = "ISO-8859-1";
  typedef void (*kernel_type) (int*, int);
  const kernel_type (*inc_kernel)(int, int) = polyglot_invoke(
    cu,
    "bindkernel",
    polyglot_from_string("increment.cubin", charset),
    polyglot_from_string(
      "c_inc_kernel(inout_arr: inout pointer sint32, n_elements: sint32)",
      charset)
  );

  const int num_elements = 1000;
  int32_t * device_array = polyglot_as_i32_array(
    polyglot_invoke(cu, "DeviceArray",
      polyglot_from_string("int", charset), num_elements));
  for (int i = 0; i < num_elements; ++i) {
    device_array[i] = i;
  }

  const int num_blocks = 32;
  const int block_size = 256;
  inc_kernel(num_blocks, block_size)(device_array, num_elements);

  for (int i = 0; i < num_elements; ++i) {
    printf("%d\n", device_array[i]);
  }
}
```

Compilation and execution of example using the Sulong's LLVM interpreter:

```bash
clang -g -O1 -c -emit-llvm -I${GRAALVM_DIR}/jre/languages/llvm/include increment.c
lli --polyglot --jvm --vm.Dtruffle.class.path.append=${GRCUDA_JAR} increment.bc
```
