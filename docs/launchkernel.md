# Polyglot Kernel Launch

## Launching prebuilt GPU Kernels from polyglot Host Languages

Existing CUDA kernels that are complied into `cubin` files
can be bound to callable objects using `bindkernel(..)`
function and subsequently from the various host languages in
the Graal/Truffle ecosystem.

### Example Kernel

Consider the following simple kernel function that takes two
device pointers, each pointing to a device-resident array of
floats. The first pointer specifies the output array, the second
the array containing the input data. The last argument indicates how
the number of elements, i.e., the capacity of both arrays.

```C++
extern "C" __global__
void inc_kernel(float* arr_out, const float* arr_in,
                size_t n_elements) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements;
       i += blockDim.x * gridDim.x) {
    arr_out[i] = arr_in[i] + 1.0f;
  }
}
```

The `extern "C"` specifier disables the C++ name mangling. If omitted,
the name of the kernel function used in the `bindkernel` function
(see below) needs to be changed to `_Z10inc_kernelPfPKfm`.

```console
$  c++filt _Z10inc_kernelPfPKfm
inc_kernel(float*, float const*, unsigned long)
```

The kernel is stored in a CUDA C/C++ source file `inc_kernel.cu` and
compiled to a cubin using the
[NVIDIA CUDA Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
`nvcc`.

```console
$  nvcc --cubin --generate-code arch=compute_70,code=sm_70 inc_kernel.cu
$
```

### <a name="bindkernel"></a> Binding Kernels in grCUDA

Kernel functions are bound to callable objects in the host
lost language. A callable is obtained using the `bindkernel(..)`
function that is built into grCUDA. The signature of the
`bindkernel(..)` function is:

```text
bindkernel(cubinFile, kernelName, signatureString)
```

The `cubinFile` argument is the path cubin file that contains the
precompiled GPU kernel.

The second argument, `kernelName` contains the name of the kernel
function, more precisely, the symbol name of the kernel function
in the cubin file. Declare the CUDA kernel as `extern "C"` to
prevent C++ name mangling. This way the `kernelName` specified
in the bind function is equal to identifier name in the CUDA
source code.

The third argument specifies the signature, i.e., the argument
types of the kernel function. The `signatureString` is a comma separated list of type names. The type names are the same as in the
[TruffleNFI language](https://github.com/oracle/graal/blob/master/truffle/src/com.oracle.truffle.nfi/src/com/oracle/truffle/nfi/impl/LibFFIType.java).
Note that parenthesis and the return type are
omitted since the return value of a kernel function is always `void`.

| NFI Type  | CUDA C/C++ Type  |
|-----------|------------------|
| `sint8`   | `char`           |
| `uint8`   | `unsigned char`  |
| `sint16`  | `short`          |
| `uint16`  | `unsigned short` |
| `sint32`  | `int`            |
| `uint32`  | `unsigned int`   |
| `sint64`  | `long`           |
| `uint64`  | `unsigned long`  |
| `float`   | `float`          |
| `double`  | `double`         |
| `pointer` | any `T*` type    |

Note that as far as the the NFI interface is concerned, pointers
are not typed, i.e., the NFI type of a kernel argument `int*` and
`float*` are both `pointer`.

**Example:**

The `bindkernel` call for the kernel function shown above is:

```C
bindkernel("inc_kernel.cubin", "inc_kernel",
           "pointer, pointer, uint64")
```

## Runtime-compilation of GPU Kernels from CUDA C/C++

grCUDA can also compile GPU kernels directly from CUDA C/C++
source code that is passed as a host-string argument to
`buildkernel(..)`. The signature of the function is:

```text
bindkernel(cudaSourceString, kernelName, signatureString)
```

The `cudaSourceString` argument contains the kernel source code
in CUDA C/C++ as a host-language string.

The second argument, `kernelName`, specifies the name of the kernel
function.

This third argument, `signatureString`, specifies signature of the
kernel function. The syntax is identical as in the corresponding
argument in the `bindkernel(..)` function.

### Example

```python
kernel_code = """
__global__
void inc_kernel(float* arr_out, const float* arr_in,
                size_t n_elements) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements;
       i += blockDim.x * gridDim.x) {
    arr_out[i] = arr_in[i] + 1.0f;
  }
}"""
buildkernel = polyglot.eval(language='grcuda', string='buildkernel')
inc_kernel = inc_kernel(kernel_source, 'inc_kernel',
                        'pointer, pointer, uint64')
```

## Launching Kernels

Once a kernel function is bound to a callable host-object,
it can be launched like a function with two argument lists (for exceptions in Ruby and Java and Ruby see below).

```test
kernel(gridsize, blocksize)(arg1, ..., argN)
```

This form is based on the triple angle brackets syntax of CUDA C/C++.

```C
kernel<<<gridsize, blocksize>>>(arg1, ..., argN)
```

The first argument list corresponds to the launch configuration, i.e.,
the kernel grid (number of blocks) and the block sizes (number of
threads per block). Two argument lists can also be partially applied.

__Example:__

```Python
gridsize = 160
blocksize = 1024
# concatentated invocation using both argument lists
kernel(gridsize, blocksize)(out_arr, in_arr, num_elements)

# kernel with launch config results in a kernel with set config
configured_kernel = kernel(gridsize, blocksize)
configured_kernel(out_arr, in_ar, num_elements)
```

grCUDA also supports 2D and 3D kernel grids that are specified
with the `dim3` in CUDA C/C++. In grCUDA `gridsize` and `blocksize`
can be integers for 1-dimensional kernels or host language sequences
of length 1, 2, or 3 (Lists or Tuples in Python, Arrays in JavaScript
and Ruby, and vectors in R)

```Python
matmult((num_block_x, num_blocks_y),
        (block_size_x, block_size_y))(matrix_A, matrix_B,
                                      n, m, k)
fft3d((nx, ny, nz), (bx, by, bz))(matrix_A, matrix_B,
                                      n, m, k)
```

Additionally, the number of bytes to allocate for shared memory
per thread block is passed as the third argument in the
kernel configuration call, as in CUDA's `<<< >>>` notation.

```Python
num_shmem_bytes = 1024
kernel_using_shmem(grid_size, block_size, num_shmem_bytes)(
                   inputA, inputB, results, N)
```

The syntax of most Graal language allows a concatenated invocation
with both argument lists. The exceptions are Ruby and Java:

In Ruby, callables need to be invoked using the `call(..)` method,
resulting in the following chained invocation:

```Ruby
kernel.call(gridsize, blocksize).call(out_arr, in_arr, num_elements)
```

For Java, the `execute()` needs to be used:

```Java
kernel.execute(gridsize, blocksize).execute(out_arr, in_arr, num_elements)
```

The current implementation of only supports synchronous kernel launches,
i.e., there is an implicit `cudaDeviceSynchronize()` after every
launch.

## Host Language Examples

### Python

```python
import polyglot

num_elements = 1000
in_array = polyglot.eval(language='grcuda',
                         string='float[{}]'.format(num_elements))
out_array = polyglot.eval(language='grcuda',
                          string='float[{}]'.format(num_elements))

# bind kernel function
inc_kernel = polyglot.eval(
    language='grcuda',
    string='bindkernel("inc_kernel.cubin", "inc_kernel", "pointer, pointer, uint64")')

gridsize = 160
blocksize = 1024
# invoke kernel as inc_kernel<<<gridsize, blocksize>>>(...)
inc_kernel(gridsize, blocksize)(out_array, in_array, num_elements)
```

### JavaScript/Node

```JavaScript
let numElements = 1000
let inArray = Polyglot.eval('grcuda', string=`float[${numElements}]`)
let outArray = Polyglot.eval('grcuda', string=`float[${numElements}]`)

// bind kernel function
incKernel = Polyglot.eval('grcuda',
    'bindkernel("inc_kernel.cubin", "inc_kernel", "pointer, pointer, uint64")')

let gridSize = 160
let blockSize = 1024
// invoke kernel as inc_kernel<<<gridsize, blocksize>>>(...)
incKernel(gridSize, blockSize)(outArray, inArray, numElements)
```

### Ruby

```ruby
num_elements = 1000
in_array = Polyglot.eval("grcuda", "float[#{num_elements}]")
out_array = Polyglot.eval("grcuda", "float[#{num_elements}]")

# bind kernel function
inc_kernel = Polyglot.eval('grcuda',
 "bindkernel(\"inc_kernel.cubin\", \"inc_kernel\", \"pointer, pointer, uint64\")")

gridsize = 160
blocksize = 1024
# invoke kernel as inc_kernel<<<gridsize, blocksize>>>(...)
inc_kernel.call(gridsize, blocksize).call(out_array, in_array, num_elements)
```

### R

```R
num_elements <- 1000
in_array <- eval.polyglot('grcuda', sprintf('float[%d]', num_elements))
out_array <- eval.polyglot('grcuda', sprintf('float[%d]', num_elements))

# bind kernel function
inc_kernel = eval.polyglot('grcuda',
  'bindkernel("inc_kernel.cubin", "inc_kernel", "pointer, pointer, uint64")')

gridsize <- 160
blocksize <- 1024
# invoke kernel as inc_kernel<<<gridsize, blocksize>>>(...)
inc_kernel(gridsize, blocksize)(out_array, in_array, num_elements)
```

### Java

```Java
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

class LaunchKernel {
  public static void main(String[] args) {
    Context polyglot = Context.newBuilder().allowAllAccess(true).build();
    int numElements = 1000;
    Value inArray = polyglot.eval("grcuda", "float[" + numElements + "]");
    Value outArray = polyglot.eval("grcuda", "float[" + numElements + "]");

    // bind kernel function
    Value incKernel = polyglot.eval("grcuda", "bindkernel(\"inc_kernel.cubin\", "+
      "\"inc_kernel\", \"pointer, pointer, uint64\")");

    int gridSize = 160;
    int blockSize = 1024;
    // invoke kernel as inc_kernel<<<gridsize, blocksize>>>(...)
    incKernel.execute(gridSize, blockSize).execute(outArray, inArray, numElements);
  }
}
```

### C through Sulong

```C
#include <stido.h>
#include <polyglot.h>

int main() {
  const int max_len = 100;
  char code_str[max_len];
  int num_elements = 1000;
  snprintf(code_str, max_len, "float[%d]", num_elements);
  void * in_array = polyglot_eval("grcuda", code_str);
  void * out_array = polyglot_eval("grcuda", code_str);

  // kernel type
  typedef void (*kernel_type) (float*, float*, int);
  // bind kernel function
  const kernel_type (*inc_kernel)(int, int) = poly_eval("grcuda",
    "bindkernel(\"inc_kernel.cubin\", \"inc_kernel\", \"pointer, pointer, uint64\")");

  const int gridsize = 160;
  const int blocksize = 1024;
  // invoke kernel as inc_kernel<<<gridsize, blocksize>>>(...)
  inc_kernel(gridsize, blocksize)(out_array, in_array, num_elements);
}
```
