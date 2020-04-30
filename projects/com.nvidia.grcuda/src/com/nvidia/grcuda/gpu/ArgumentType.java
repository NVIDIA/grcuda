package com.nvidia.grcuda.gpu;

/**
 * Defines the types of arguments that can be specified in a kernel NFI signature;
 */
public enum ArgumentType {
    POINTER,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64;
}