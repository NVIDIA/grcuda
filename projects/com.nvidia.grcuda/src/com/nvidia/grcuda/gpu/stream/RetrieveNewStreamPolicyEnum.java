package com.nvidia.grcuda.gpu.stream;

public enum RetrieveNewStreamPolicyEnum {
    FIFO("fifo"),
    ALWAYS_NEW("always_new");

    private final String name;

    RetrieveNewStreamPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
