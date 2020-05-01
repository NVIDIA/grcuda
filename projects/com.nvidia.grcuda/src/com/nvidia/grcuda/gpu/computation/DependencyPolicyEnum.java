package com.nvidia.grcuda.gpu.computation;

public enum DependencyPolicyEnum {
    DEFAULT("default"),
    WITH_CONST("with_const");

    private final String name;

    DependencyPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
