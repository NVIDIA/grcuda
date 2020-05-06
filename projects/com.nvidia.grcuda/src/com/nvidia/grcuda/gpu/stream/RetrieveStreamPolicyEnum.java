package com.nvidia.grcuda.gpu.stream;

public enum RetrieveStreamPolicyEnum {
    LIFO("lifo"),
    ALWAYS_NEW("always_new");

    private final String name;

    RetrieveStreamPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
