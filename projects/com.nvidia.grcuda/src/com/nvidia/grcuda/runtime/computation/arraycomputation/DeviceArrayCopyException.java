package com.nvidia.grcuda.runtime.computation.arraycomputation;

import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.nodes.Node;

public final class DeviceArrayCopyException extends AbstractTruffleException {
    private static final long serialVersionUID = 8614211550329856579L;

    public DeviceArrayCopyException(String message) {
        this(message, null);
    }

    public DeviceArrayCopyException(String message, Node node) {
        super(message, node);
    }
}
