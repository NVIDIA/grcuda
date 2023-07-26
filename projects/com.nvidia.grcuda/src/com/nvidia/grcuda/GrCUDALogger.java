package com.nvidia.grcuda;

import com.oracle.truffle.api.TruffleLogger;

public class GrCUDALogger {

    public static final String DEFAULT_LOGGER_LEVEL= "INFO";

    public static final String GRCUDA_LOGGER = "com.nvidia.grcuda";

    public static final String CUDALIBRARIES_LOGGER = "com.nvidia.grcuda.cudalibraries";

    public static final String FUNCTIONS_LOGGER = "com.nvidia.grcuda.functions";

    public static final String NODES_LOGGER = "com.nvidia.grcuda.nodes";

    public static final String PARSER_LOGGER = "com.nvidia.grcuda.parser";

    public static final String RUNTIME_LOGGER = "com.nvidia.grcuda.runtime";

    public static final String ARRAY_LOGGER = "com.nvidia.grcuda.runtime.array";

    public static final String COMPUTATION_LOGGER = "com.nvidia.grcuda.runtime.computation";

    public static final String EXECUTIONCONTEXT_LOGGER = "com.nvidia.grcuda.runtime.executioncontext";

    public static final String STREAM_LOGGER = "com.nvidia.grcuda.runtime.stream";

    public static TruffleLogger getLogger(String name) {
        return TruffleLogger.getLogger(GrCUDALanguage.ID, name);
    }
}
