# Extending GrCUDA with a dynamic computational DAG

The main idea is to **represent GrCUDA computations as vertices of a DAG**, connected using their dependencies (e.g. the output of a kernel is used as input in another one).
 * The DAG allows scheduling parallel computations on different streams and avoid synchronization when not necessary
 * See `projects/resources/python/examples` and `projects/resources/python/benchmark/bench` for simple examples of how this technique can be useful
 
**Differences w.r.t. existing techniques** (e.g. TensorFlow or [CUDA Graphs](https://devblogs.nvidia.com/cuda-graphs/)):
 1. The DAG creation is automatic, instead of being built by the user
 2. The DAG is built at runtime, not at compile time or eagerly. This means that we don't have to worry about the control flow of the host program, but only about data dependencies, 
 as we dynamically add and schedule new vertices/computations as the user provides them. We can also collect profiling information and adjust the DAG creation based on that 
 (e.g. how many CUDA streams we need, or how large each GPU block should be)

**How it works, in a few words**    
 * The class `GpuExecutionContext` tracks GPU computational elements (e.g. `kernels`) declarations and invocations
 * When a new computation is created, or when it is called, it notifies `GpuExecutionContext` so that it updates the `DAG` by computing the data dependencies of the new computation
 * `GpuExecutionContext` uses the DAG to understand if the new computation can start immediately, or it must wait for other computations to finish
 * Different computations are overlapped using different CUDA streams, assigned by the `GrCUDAStreamManager` based on dependencies and free resources
 * Computations on the GPU are asynchronous and are scheduled on streams without explicit synchronization points, as CUDA guarantee that computations are stream-ordered
 * Synchronziation between streams happens with CUDA events, without blocking the host CPU thread
 * If a computation is done by the CPU (e.g. an array read) we synchronize only the necessary streams, and the host is blocked until the required data is available 
 
## What's already there

* The DAG supports kernel invocation, and array accesses (both `DeviceArray` and `MultiDimDeviceArray`)
    * Kernels are executed in parallel, on different streams, whenever possible
* **Main classes used by the scheduler**
    1. `GpuExecutionContext`: takes care of scheduling and executing computations, it is the director of the orchestration and manages the DAG
    2. `GrCUDAComputationalElement`: abstract class that wraps GrCUDA computations, e.g. kernel executions and array accesses. 
    It provides `GpuExecutionContext` with functions used to compute dependencies or decide if the computation must be done synchronously (e.g. array accesses)
    3. `ExecutionDAG`: the DAG representing the dependencies between computations, it is composed of vertices that wrap each `GrCUDAComputationalElement`
    4. `GrCUDAStreamManager`: class that handles the creation and the assignment of streams to kernels, and the synchronization between different streams or the host thread
* **Basic execution flow**
    1. The host language (i.e. the user) calls an `InteropLibrary` object that can be associated to a `GrCUDAComputationalElement`, e.g. a kernel execution or an array access
    2. A new `GrCUDAComputationalElement` is created and registered to the `GpuExecutionContext`, to represent the computation
    3. `GpuExecutionContext` adds the computation to the DAG and computes its dependencies
    4. Based on the dependencies, the `GpuExecutionContext` associates a stream to the computation through `GrCUDAStreamManager`
    5. `GpuExecutionContext` executes the computation on the chosen stream, performing synchronization if necessary
    * GPU computations do not require synchronization w.r.t. previous computations on the stream where they executed, as CUDA guarantees stream-ordered execution.
     CUDA streams are synchronized with (asynchronous) CUDA events, without blocking the host. 
     CPU computations that require a GPU result are synchronized with `cudaStreamSynchronize` only on the necessary streams
    6. In case of subsequent array accesses, we skip the scheduling part as accesses are synchronous, and minimize overheads
    7. From the point of view of GrCUDA, asynchronous GPU computations are considered **active** until the CPU requires the result of either them or their children.
     Active computations are used in dependency computations (unless all their parameters have been *covered* by children) and to determine which streams are free.
* The CUDA stream interface has been added to GrCUDA, and is accessible by the users (not recommended, but possible)
    * Users can create/destroy streams, and assign streams to kernels
    * The CUDA events API is also available
    * The `cudaStreamAttachMemAsync` is also exposed, to exclusively associate a managed memory array to a given stream. 
    This is used, on Pre-Pascal GPUs, to access arrays on CPU while a kernel is using other arrays on GPU
* Most of the new code is unit-tested and integration-tested, and there is a Python benchmarking suite to measure execution time with different settings
    * For example, the file `projects/resources/python/benchmark/bench/bench_8` is a fairly complex image processing pipeline that automatically manages up to 4 different streams
    Compared to sequential scheduling, we are up to **2.5x faster**!
* **Streams** are managed internally by the GrCUDA runtime: we keep track of existing streams that are currently empty, and schedule computations on them in a FIFO order.
 New streams are created only if no existing stream is available
* **Read-only** input arguments can be specified with the `const` keyword; they will be ignored in the dependency computations if possible:
 for example, if there are 2 kernels that use the same read-only input array, they will be executed concurrently 

* **Current limitations**
    1. ~~Dependency computation does not consider disjoint parameter subsets.~~ **Now available!**
     Consider 3 kernels, `K1(X, Y)`, `K2(X)`, `K3(Y)`: `K2` and `K3` are both depending on `K1`, but are using different inputs, and can run in parallel.
    2. ~~Synchronization happens the main execution thread.~~ **We now support fully asynchronous GPU execution on a single CPU thread thanks to CUDA events**
     in the example before, calling `K2` requires to sync on the stream used by `K1`, and `K3` starts only after `K2` has started.
      If `Y` was read-only in `K1`, this wait would have been unnecessary
    3. **Scalar values are not considered for dependencies**: they are read-only when used as input, but there could be output-input dependencies in library functions with scalar output ([API Design, point 4](#api-design)) 
    4. Read-only arguments are visible on the **default stream**, instead of having their visibility limited to the stream where they are used.
     This makes the arguments visible to kernels running on different streams, but accesses by the CPU to these arguments require full device synchronization.
     This limitation is likely to occur only on Pre-Pascal devices, however
    5. **Library functions are not considered for asynchronous execution**: not all library functions expose a stream interface for asynchronous execution, and they are currently ignored by the DAG scheduler. 
    We need to add, at the very least, a synchronization point before synchronous library functions

## Open questions

### Questions on API design (i.e. how do we provide the best user experience)

1. ~~How to understand if a parameter is read-only? ([API Design, point 4](#api-design))~~ We can use the `const` keyword in the computation signature
2. How do we track scalar values in library function outputs? ([API Design, point 5](#api-design))
3. How can user specify options cleanly? ([API Design, point 2](#api-design))
    * Using only context startup options is limiting, but it simplify the problem (we don't have to worry about changing how the DAG is built at runtime)
    * If we want provide more flexibility, we can add functions to the DSL, but that's not very clean

### Questions on internal development (i.e. how do we do something in the most powerful/flexible/efficient way)

1. How to handle library functions? They usually have no stream options ([API Design, point 6](#api-design))
2. How to handle pre-registered libraries and external functions? Same problem as question 1

### Other questions (e.g. things I don't understand about GrCUDA/GraalVM/Truffle)

1. What are `map` and `shred` functions? Are they exposed to the outside?
2. When doing unit-testing, can we access internal data structures of the guest language (e.g. to monitor the state of the DAG)
3. In Graalpython, can we create new guest polyglot contexts at runtime, with user-specified options?

*** 

## Detailed development notes

### API Design
    
Dependencies are inferred automatically, instead of being manually specified by the user using handles
 1. Automatic dependency inferring is more interesting from a research perspective, and *cleaner* for end-users
     * One option is to perform synchronization *white-listing*: have explicit sync points after every kernel call, and remove dependencies if possible.
      **Pro:** it should be better for guaranteeing correctness. **Cons:** finding if we *do not* have a dependency is more complex than finding if we have one
     * The other option is *black-listing*, i.e. do not have any sync point and add them if a dependency is found.
      This is the option currently used: it is simpler, faster, and provides identical results to the other approach
 2. The API needs ways to modify the scheduling policy, if desired (e.g. go back to fully synchronized execution)
     * Context startup option? Easy, but cannot be modified
     * Expose a function in the GrCUDA DSL? More flexibility, but changing options using the DSL is not very clean
 3. ~~How to handle CPU control flow? In GrCUDA we are not aware of `if` and `for` loops on the host side~~
     * The DAG is built dynamically: we need to update it as we receive scheduling orders, and decide if we can execute or not. We don't care about the original control flow
 4. How do we identify if a **parameter is read-only**? If two kernels use the same parameter but only read from it, they can execute in parallel
     * This is not trivial: LLVM can understand, for example, if a scalar value is read-only, but doing that with an array is not always possible
     * Users might have to specify which parameters are read-only in the kernel signature, which is still better than using explicit handles
     * For now, we let programmers manually specify read-only array arguments using the `const` keyword, as done in `CUDA`
 5. How do we handle scalar values? We could also have dependencies due to scalar values (e.g. a computation is started only if the error in the next iteration is above a threashold)
     * Currently, only reads from `DeviceArray` (and similar) return scalar values, and they must be done synchronously, as the result is immediately exposed to the guest language. 
     * Array reads (and writes) are done synchronously by the host, and we guarantee that no kernel that uses the affected array is running
     * Kernels do not return scalar values, and scalar outputs are stored in a size-1 array (which we can treat as any other array)
     * Then the programmer can pass the size-1 array to another computation (handled like any array), or extract the value with an array read that triggers synchronization
     * Scalar values are only problematic when considering library functions that return them
     * One idea could be to *box* scalar values with Truffle nodes and store the actual value using a `Future`.
     If the user reads or writes the value, we wait for the GPU computation to end. Then the scalar value can be unboxed to avoid further overheads.
     * But running library functions on streams is problematic (see problem 6), so this solution might not be required
 6. Library functions: library functions are more complex to handle as they could also have code running on the host side.
    * They also do not expose streams, so it could be difficult to pipeline them
    * In some cases they might expose streams in the signature, we can probably find them by parsing the signature
    * They can also return scalars (see problem 5)
    * If we run them on threads, we parallelize at least the CPU side

### What is a computational element in GrCUDA?

`bindkernel`, `buildkernel` functions create a `Kernel` object that contains information about the signature and code
 * `Kernel` is an executable `InteropLibrary` class that creates a `ConfiguredKernel` that contains information about the number of blocks, shared memory size, etc...
    * Kernel arguments are provided to `ConfiguredKernel` when it's executed, although they are also passed to the corresponding `Kernel`
        
`DeviceArray` accesses can be done in any point, and are not represented as kernels (as they happen on CPU side, using managed memory)
 * If a `DeviceArray` access happens between two kernels, we must keep the sync point
 * Similar considerations for `MultiDimDeviceArray`. We don't need to deal with the outer dimensions, as only the innermost level accesses managed memory
 * Accesses to managed memory are added to the DAG only if they require synchronization. 
 If an access follows another access (e.g. when initializing an array) we can skip the scheduling and execute it immediately, without scheduling overhead
 
Pre-registered libraries, such as RAPIDS, can be called like `dbscan_func = polyglot.eval(language="grcuda", string="ML::cumlDpDbscanFit")`
 * They are added to a namespace just like `buildkernel`
 * They are retrieved using directly a `CallNode`, so we need to observe that too
 * They are called accessing the `CUMLRegistry`, and other registries, as they aren't kernels, but `ExternalFunctions`
 * `ExternalFunctions` are callable, and arguments are passed directly to them
    
Library functions (non-kernels) can also be loaded, using `BindFunction`
 * This loads the function using NFI, and returns a callable object
 * They can also return scalar values (see [API Design, point 6](#api-design))

Invocation to computational elements are wrapped in classes that extend a generic `GrCUDAComputationalElement`.
`GrCUDAComputationalElement` is used to build the vertices of the DAG and exposes interfaces to compute data dependencies with other `GrCUDAComputationalElements` and to schedule the computation
 
### Other notes on GrCUDA architecture

These notes relate to the structure of the original GrCUDA repository. You can skip them if you are already familiar with it!

The `nodes` package contains the basic Truffle nodes that define the language
 * Not relevant at the moment, as we can deal with already-parsed functions (e.g. `buildkernel`) and `InteropLibrary` objects
 * But it might be required to add nodes to handle scalar values
    
The `function` package contains functions that can be invoked through the DSL, such as `buildkernel`
 * We might want to add some function to enable the user to change the runtime behaviour ()

`Namespace` handling: the `Namespace` class maintains a tree of functions (e.g. `buildkernel`) and other namespaces (e.g. `ML`)
 * When `CallNode` is executed, we look for a function whose name matches the identifier of the `CallNode`
 * If a function has a namespace, like `ML::cumlDpDbscanFit`, it is decomposed in multiple pieces (`ML` and `cumlDpDbscanFit`), and it is retrieved with a tree visit in the namespace
 * Additional namespaces are created in `GrCUDAContext`, and a registry like `CUMLRegistry` adds functions to the namespace
     * Each function in the registry is added to the namespace as an `ExternalFunction` 
     

