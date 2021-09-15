package com.nvidia.grcuda.runtime.stream;

public enum RetrieveNewStreamPolicyEnum {
    FIFO("fifo"),
    ALWAYS_NEW("always-new");

    private final String name;

    RetrieveNewStreamPolicyEnum(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
