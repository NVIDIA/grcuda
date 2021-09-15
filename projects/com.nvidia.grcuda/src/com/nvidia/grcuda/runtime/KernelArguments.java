package com.nvidia.grcuda.runtime;

import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class KernelArguments implements Closeable {

    private final Object[] originalArgs;
    /**
     * Associate each input object to the characteristics of its argument, such as its type and if it's constant;
     */
    private final List<ComputationArgumentWithValue> kernelArgumentWithValues = new ArrayList<>();
    private final UnsafeHelper.PointerArray argumentArray;
    private final ArrayList<Closeable> argumentValues = new ArrayList<>();

    public KernelArguments(Object[] args, ComputationArgument[] kernelArgumentList) {
        this.originalArgs = args;
        this.argumentArray = UnsafeHelper.createPointerArray(args.length);
        assert(args.length == kernelArgumentList.length);
        // Initialize the list of arguments and object references;
        for (int i = 0; i < args.length; i++) {
            kernelArgumentWithValues.add(new ComputationArgumentWithValue(kernelArgumentList[i], args[i]));
        }
    }

    public void setArgument(int argIdx, UnsafeHelper.MemoryObject obj) {
        argumentArray.setValueAt(argIdx, obj.getAddress());
        argumentValues.add(obj);
    }

    long getPointer() {
        return argumentArray.getAddress();
    }

    public Object[] getOriginalArgs() {
        return originalArgs;
    }

    public Object getOriginalArg(int index) {
        return originalArgs[index];
    }

    public List<ComputationArgumentWithValue> getKernelArgumentWithValues() {
        return kernelArgumentWithValues;
    }

    @Override
    public String toString() {
        return "KernelArgs=" + Arrays.toString(originalArgs);
    }

    @Override
    public void close() {
        this.argumentArray.close();
        for (Closeable c : argumentValues) {
            try {
                c.close();
            } catch (IOException e) {
                /* ignored */
            }
        }
    }
}
