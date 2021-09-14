package com.nvidia.grcuda.test;

import java.util.logging.Handler;
import java.util.logging.LogRecord;

public final class TestLogHandler extends Handler {
    private volatile boolean closed;

    TestLogHandler() {
    }

    @Override
    public void publish(LogRecord record) {
        if (closed) {
            throw new IllegalStateException("Closed handler");
        }
        System.out.println("[" + record.getLoggerName() + "] " + record.getLevel() + ": " + record.getMessage());
    }

    @Override
    public void flush() {
        if (closed) {
            throw new IllegalStateException("Closed handler");
        }
    }

    @Override
    public void close() throws SecurityException {
        closed = true;
    }
}