    
    
# Extending GrCUDA with a dynamic computational DAG

The main idea is to **represent GrCUDA computations as vertices of a DAG**, connected using their dependencies (e.g. the output of a kernel is used as input in another one).
 * The DAG allows scheduling parallel computations on different streams and avoid synchronization when not necessary
 * See `projects/resources/python/examples/pipeline_1.py` for a simple example of how it can be useful
 
Differences w.r.t. existing techniques (e.g. TensorFlow or [CUDA Graphs](https://devblogs.nvidia.com/cuda-graphs/)):
 1. The DAG creation is automatic, instead of being built by the user
 2. The DAG is built at runtime, not at compile time. This means that we don't have to worry about the control flow of the host program, but only about data dependencies, 
 as we dynamically add and schedule new vertices/computations as the user provides them. We can also collect profiling information and adjust the DAG creation based on that (e.g. how many CUDA streams we need)
    
We need a class (e.g. `GpuExecutionContext`) that tracks GPU computational elements (e.g. `kernels`) declarations and invocations
 * When a new computation is created, or when it is called, notify `GpuExecutionContext` so that it updates the `DAG`
 * Different computations are overlapped using different CUDA streams
 * If a computation requires one or more computations to finish before starting, it will have to wait by using a synchronization point on the right CUDA stream(s)
 * Otherwise it will execute immediately 
 
 
## API Design 
    
Dependencies can be explicitely specified by the user using handles, or we can try inferring dependencies automatically
 1. Automatic dependency inferring is more interesting from a research perspective, and *cleaner* for end-users
     * By default, we need to have explicit sync points after every kernel call
     * Automatic dependency inferring will remove some sync points if possible
     * It must be a *white-listing* process, as we need to guarantee correctness
 2. The API needs ways to modify/turn off this policy, if desired
     * Startup option? Easy, but cannot be modified
     * Expose a function in the GrCUDA DSL? More flexibility, but changing options using the DSL is not very clean
 3. How to handle CPU control flow? In GrCUDA we are not aware of `if` and `for` loops on the host side
     * The DAG is built dynamically: we need to update it as we receive scheduling orders, and decide if we can execute or not. We don't care about the original control flow
 4. How do we identify if a **parameter is read-only**? If two kernels use the same parameter but only read from it, they can execute in parallel
     * This is not trivial: LLVM can understand, for example, if a scalar value is read-only, but doing that with an array is not always possible
     * Users might have to specify which parameters are read-only in the kernel signature, which is probably better than using explicit handles
 5. How do we handle scalar values? We could also have dependencies due to scalar values (e.g. a computation is started only if the error in the next iteration is above a threashold)
    * Scalars are returned by copy, so we cannot keep track of their dependencies. Input scalars are read-only by definition, but what about output scalars?
    * Currently kernels cannot return scalar values, and scalar outputs are stored in a size-1 array (which we can treat as any other array)
    * But library functions can return scalars! 
    * One idea could be to *box* scalar values with Truffle nodes and store the actual value using a `Future`.
     If the user reads or writes the value, we wait for the GPU computation to end. Then the scalar value can be unboxed to avoid further overheads. 
 6. Libraries functions: library functions are more complex to handle as they could also have code running on the host side.
    * They also do not expose streams, so it could be difficult to pipeline them
    * In some cases they might expose streams in the signature, we can probably find them by parsing the signature
    * They can also return scalars (see problem 5)


## What is a computational element in GrCUDA?

`bindkernel`, `buildkernel` functions create a `Kernel` object that contains information about the signature and code
 * `Kernel` is an executable `InteropLibrary` class that creates a `ConfiguredKernel` that contains information about the number of blocks, shared memory size, etc...
    * Kernel arguments are provided to `ConfiguredKernel` when it's executed, although they are also passed to the corresponding `Kernel`
        
`DeviceArray` accesses can be done in any point, and are not represented as kernels (as they happen on CPU side, using managed memory)
 * If a `DeviceArray` access happens between two kernels, we must keep the sync point
 * Similar considerations for `MultiDimDeviceArray`. We don't need to deal with inner dimensions, the top-level access is enough to keep a sync point
 
Pre-registered libraries, such as RAPIDS, can be called like `dbscan_func = polyglot.eval(language="grcuda", string="ML::cumlDpDbscanFit")`
 * They are added to a namespace just like `buildkernel`
 * They are retrieved using directly a `CallNode`, so we need to observe that too
 * They are called accessing the `CUMLRegistry`, and other registries, as they aren't kernels, but `ExternalFunctions`
 * `ExternalFunctions` are callable, and arguments are passed directly to them
    
Library functions (non-kernels) can also be loaded, using `BindFunction`
 * This loads the function using NFI, and returns a callable object
 * They can also return scalar values (see [API Design, point 6](#api-design))

Other stuff? E.g. `map` and `shred`, currently not documented

Invocation to computational elements are wrapped in classes that extend a generic `GrCUDAComputationalElement`, 
which is used to build the vertices of the DAG and exposes interfaces to compute data dependencies with other `GrCUDAComputationalElements` and to schedule the computation
 
# Other notes on GrCUDA architecture

The `nodes` package contains the basic Truffle nodes that define the language
 * Not immediately relevant to this project, we can deal with already-parsed functions (e.g. `buildkernel`) and `InteropLibrary` objects
 * But it might be required to add nodes to handle scalar values
    
The `function` package contains functions that can be invoked through the DSL, such as `buildkernel`
 * We might want to add some function to change the runtime behaviour at runtime

`Namespace` handling: the `Namespace` class maintains a tree of functions (e.g. `buildkernel`) and other namespaces (e.g. `ML`)
 * When `CallNode` is executed, we look for a function whose name matches the identifier of the `CallNode`
 * If a function has a namespace, like `ML::cumlDpDbscanFit`, it is decomposed in multiple pieces (`ML` and `cumlDpDbscanFit`), and it is retrieved with a tree visit in the namespace
 * Additional namespaces are created in `GrCUDAContext`, and a registry like `CUMLRegistry` adds functions to the namespace
     * Each function in the registry is added to the namespace as an `ExternalFunction` 
     
## Open questions

1. What are `map` and `shred` functions? Are they exposed to the outside?
2. How to handle library functions? Wrapping them with a `GrCUDAComputationalElement`?
3. How to handle pre-registered libraries and external functions? Wrapping them with a `GrCUDAComputationalElement`?
4. How do we modify kernel calls to add/remove sync points?
5. How do we execute kernels in parallel? We need to have streams, and a stream manager
6. How do we monitor accesses to `DeviceArrays` to preserve sync points?    
7. How do we track scalar values?
8. How to understand if a parameter is read-only?
9. When doing unit-testing, can we access internal data structures of the guest language (e.g. to monitor the state of the DAG)
