package com.nvidia.grcuda.runtime.stream.policy;

public enum DeviceSelectionPolicyEnum {
    SINGLE_GPU("single-gpu"),
    ROUND_ROBIN("round-robin"),
    STREAM_AWARE("stream-aware"),
    MIN_TRANSFER_SIZE("min-transfer-size"),
    MINMIN_TRANSFER_TIME("minmin-transfer-time"),
    MINMAX_TRANSFER_TIME("minmax-transfer-time");

    private final String name;

    DeviceSelectionPolicyEnum(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }
}