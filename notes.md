# Notes on GrCUDA architecture



# Notes on GrCUDA extensions

* Idea: keep a computational DAG that connects GPU computations expressed in GrCUDA and their dependencies
    * Then use the DAG to schedule parallel computations on different streams and avoid synchronization when not necessary
    * See `projects/resources/python/examples/pipeline_1.py` for an example
    
* We need a class (e.g. `GpuExecutionContext`) that tracks GPU computational elements (i.e. `kernels`) declarations and invocations
    * Ideally, when a new kernel is created, or when it is called, update the context with the new information
    * Then the context will decide whether to execute it immediately or add a sync point (this could also be done in a different class, design is WIP)

* What is a computational element in GrCUDA?
    * `bindkernel`, `buildkernel` functions create a `Kernel` object that contains information about the signature and code
    * `Kernel` is an executable `InteropLibrary` class that creates a `ConfiguredKernel` that contains information about the number of blocks, shared memory size, etc...
        * `ConfiguredKernel` could also contain *dependency handles* specified by the programmer, or created internally
        * Pro: easy to extend. Cons: it breaks the similarity with CUDA <<< ... >>> syntax
        * Alternative: `ConfiguredKernel` can also be called with a single `Handle` parameter, and it will return a new `ConfiguredKernelWithHandle` that can be executed with the kernel args
        * Kernel arguments are provided to `ConfiguredKernel` when it's executed, although they are also passed to the corresponding `Kernel`
    * `DeviceArray` accesses can be done in any point, and are not represented as kernels (as they happen on CPU side, using managed memory)
        * If a `DeviceArray` access happens between two kernels, we must keep the sync point
    
* API: we can have explicit handles, or infer dependencies automatically
    * Automatic dependency inferring is more interesting from a research perspective, and **cleaner** for end-users
    * But more complex to implement, clearly
    * By default, we need to have explicit sync points after every kernel call
    * Automatic dependency inferring will remove some sync points if possible
    * It must be a **white-listing** process, as we need to guarantee correctness
    * The API needs ways to modify/turn off this policy, if desired
        * Startup option? Easy, but cannot be modified
        * Expose a function in the GrCUDA DSL? More flexibility, but changing options using the DSL is not very clean
    * Problem: how to handle CPU control flow? In GrCUDA we are not aware of `if` and `for` loops on the host side
        * The DAG cannot be build statically, we need to update it as we receive scheduling orders, and decide if we can execute or not
       