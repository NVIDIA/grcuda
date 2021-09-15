package com.nvidia.grcuda.runtime.computation;

import com.nvidia.grcuda.functions.Function;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.DefaultStream;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.List;
import java.util.stream.Collectors;

import static com.nvidia.grcuda.functions.Function.INTEROP;

/**
 * Computational element that wraps calls to CUDA libraries such as cuBLAS or cuML.
 */
public class CUDALibraryExecution extends GrCUDAComputationalElement {

    private final Function nfiFunction;
    private final Object[] argsWithHandle;

    public CUDALibraryExecution(AbstractGrCUDAExecutionContext context, Function nfiFunction, List<ComputationArgumentWithValue> args) {
        super(context, new CUDALibraryExecutionInitializer(args));
        this.nfiFunction = nfiFunction;

        // Array of [libraryHandle + arguments], required by CUDA libraries for execution;
        this.argsWithHandle = new Object[args.size()];
        for (int i = 0; i < args.size(); i++) {
            argsWithHandle[i] = args.get(i).getArgumentValue();
        }
    }

    @Override
    public boolean canUseStream() { return false; }

    // TODO: See note in parent class;
//    @Override
//    public boolean mustUseDefaultStream() { return true; }

    @Override
    public Object execute() throws UnsupportedTypeException {
        // Execution happens on the default stream;
        Object result = null;
        try {
            result = INTEROP.execute(this.nfiFunction, this.argsWithHandle);
        } catch (ArityException | UnsupportedMessageException e) {
            System.out.println("error in execution of cuBLAS function");
            e.printStackTrace();
        }
        // Synchronize only the default stream;
        this.grCUDAExecutionContext.getCudaRuntime().cudaStreamSynchronize(DefaultStream.get());
        this.setComputationFinished();
        return result;
    }

    static class CUDALibraryExecutionInitializer implements InitializeDependencyList {
        private final List<ComputationArgumentWithValue> args;

        CUDALibraryExecutionInitializer(List<ComputationArgumentWithValue> args) {
            this.args = args;
        }

        @Override
        public List<ComputationArgumentWithValue> initialize() {
            // Consider only arrays as dependencies;
            // FIXME: should the library handle be considered a dependency?
            //  The CUDA documentation is not clear on whether you can have concurrent computations with the same handle;
            return this.args.stream().filter(ComputationArgument::isArray).collect(Collectors.toList());
        }
    }
}
