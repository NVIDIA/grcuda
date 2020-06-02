package com.nvidia.grcuda.gpu.executioncontext;

public enum ExecutionPolicyEnum {
    SYNC("sync"),
    DEFAULT("default");

    private final String name;

    ExecutionPolicyEnum(String name) {
        this.name = name;
    }

    public final String getName() {
        return name;
    }
}
