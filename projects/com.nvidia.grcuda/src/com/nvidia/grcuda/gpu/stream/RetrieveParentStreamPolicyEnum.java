package com.nvidia.grcuda.gpu.stream;

public enum RetrieveParentStreamPolicyEnum {
    SAME_AS_PARENT("same-as-parent"),
    DISJOINT("disjoint");

    private final String name;

    RetrieveParentStreamPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
