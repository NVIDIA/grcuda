package com.nvidia.grcuda.cudalibraries;

import com.nvidia.grcuda.ComputationArgument;
import com.nvidia.grcuda.ComputationArgumentWithValue;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.functions.Function;
import com.oracle.truffle.api.CompilerDirectives;

import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper class to CUDA library functions. It holds the signature of the function being wrapped,
 * and creates {@link com.nvidia.grcuda.ComputationArgument} for the signature and inputs;
 */
public abstract class CUDALibraryFunction extends Function {

    private final List<ComputationArgument> computationArguments;

    /**
     * Constructor, it takes the name of the wrapped function and its NFI signature,
     * and creates a list of {@link com.nvidia.grcuda.ComputationArgument} from it;
     * @param name name of the function
     * @param nfiSignature NFI signature of the function
     */
    protected CUDALibraryFunction(String name, String nfiSignature) {
        super(name);
        // Create the list of computation arguments;
        try {
            this.computationArguments = ComputationArgument.parseParameterSignature(nfiSignature);
        } catch (TypeException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(e.getMessage());
        }
    }

    /**
     * Given a list of inputs, map each signature argument to the corresponding input.
     * Assume that inputs are given in the same order as specified by the signature.
     * In this case, provide a pointer to the CUDA library handle, which is stored as first parameter of the argument list
     * @param args list of inputs
     * @param libraryHandle pointer to the native object used as CUDA library handle
     * @return list of inputs mapped to signature elements, used to compute dependencies
     */
    public List<ComputationArgumentWithValue> createComputationArgumentWithValueList(Object[] args, Long libraryHandle) {
        ArrayList<ComputationArgumentWithValue> argumentsWithValue = new ArrayList<>();
        // Set the library handle;
        argumentsWithValue.add(new ComputationArgumentWithValue(this.computationArguments.get(0), libraryHandle));
        // Set the other arguments;
        for (int i = 0; i < args.length; i++) {
            argumentsWithValue.add(new ComputationArgumentWithValue(this.computationArguments.get(i + 1), args[i]));
        }
        return argumentsWithValue;
    }
}

