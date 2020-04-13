# Notes on GrCUDA architecture

* The `nodes` package contains the basic Truffle nodes that define the language
    * Not relevant to our goal, we can deal with already-parsed functions (e.g. `buildkernel`) and `InteropLibrary` objects
* The `function` package contains functions that can be invoked through the DSL, such as `buildkernel`
    * We might want to add observers/callbacks in some of these functions, or do it directly in the `Kernel` constructor

* `Namespace` handling: the `Namespace` class maintains a tree of functions (e.g. `buildkernel`) and other namespaces (e.g. `ML`)
    * When `CallNode` is executed, we look for a function whose name matches the identifier of the `CallNode`
    * If a function has a namespace, like `ML::cumlDpDbscanFit`, it is decomposed in multiple pieces (`ML` and `cumlDpDbscanFit`), and it is retrieved with a tree visit in the namespace
    * Additional namespaces are created in `GrCUDAContext`, and a registry like `CUMLRegistry` adds functions to the namespace
        * Each function in the registry is added to the namespace as an `ExternalFunction`
    
# Notes on GrCUDA extensions

* Idea: keep a computational DAG that connects GPU computations expressed in GrCUDA and their dependencies
    * Then use the DAG to schedule parallel computations on different streams and avoid synchronization when not necessary
    * See `projects/resources/python/examples/pipeline_1.py` for an example
    
* We need a class (e.g. `GpuExecutionContext`) that tracks GPU computational elements (i.e. `kernels`) declarations and invocations
    * Ideally, when a new kernel is created, or when it is called, update the context with the new information
    * Then the context will decide whether to execute it immediately or add a sync point (this could also be done in a different class, design is WIP)

## What is a computational element in GrCUDA?

`bindkernel`, `buildkernel` functions create a `Kernel` object that contains information about the signature and code
    * `Kernel` is an executable `InteropLibrary` class that creates a `ConfiguredKernel` that contains information about the number of blocks, shared memory size, etc...
        * `ConfiguredKernel` could also contain *dependency handles* specified by the programmer, or created internally
        * Pro: easy to extend. Cons: it breaks the similarity with CUDA <<< ... >>> syntax
        * Alternative: `ConfiguredKernel` can also be called with a single `Handle` parameter, and it will return a new `ConfiguredKernelWithHandle` that can be executed with the kernel args
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
    * It is not clear how to handle calls to the callable object. It might be required to wrap it into a custom Truffle Node
Other stuff? E.g. `map` and `shred`, currently not documented
 
## API Design 
    
We can have explicit handles, or infer dependencies automatically
    * Automatic dependency inferring is more interesting from a research perspective, and *cleaner* for end-users
    * But more complex to implement, clearly
    * By default, we need to have explicit sync points after every kernel call
    * Automatic dependency inferring will remove some sync points if possible
    * It must be a *white-listing* process, as we need to guarantee correctness
    * The API needs ways to modify/turn off this policy, if desired
        * Startup option? Easy, but cannot be modified
        * Expose a function in the GrCUDA DSL? More flexibility, but changing options using the DSL is not very clean
    * Problem: how to handle CPU control flow? In GrCUDA we are not aware of `if` and `for` loops on the host side
        * The DAG cannot be build statically, we need to update it as we receive scheduling orders, and decide if we can execute or not

## Open questions

1. What are `map` and `shred` functions? Are they exposed to the outside?
2. How to handle library functions? Wrapping them with a Truffle Node?
3. How to handle pre-registered libraries and external functions? Can we maybe use the same wrapping trick as library functions?
4. How do we modify kernel calls to add/remove sync points?
5. How do we execute kernels in parallel? We need to have streams
6. How do we monitor accesses to `DeviceArrays` to preserve sync points?    