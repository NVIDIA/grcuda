package com.nvidia.grcuda.gpu;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;

public final class KernelArguments implements Closeable {

    private final Object[] originalArgs;
    private final UnsafeHelper.PointerArray argumentArray;
    private final ArrayList<Closeable> argumentValues = new ArrayList<>();

    public KernelArguments(Object[] args) {
        this.originalArgs = args;
        this.argumentArray = UnsafeHelper.createPointerArray(args.length);
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
