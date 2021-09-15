package com.nvidia.grcuda.runtime.stream;

import java.util.Collection;

/**
 * This abstract class defines how a {@link GrCUDAStreamManager}
 * will assign a {@link CUDAStream} to a {@link com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement}
 * that has no dependency on active computations.
 * For example, it could create a new stream or provide an existing stream that is currently not used;
 */
public abstract class RetrieveNewStream {
    abstract CUDAStream retrieve();

    /**
     * Initialize the class with the provided stream,
     * for example a new stream that can be provided by {@link RetrieveNewStream#retrieve()}
     * @param stream a stream that should be associated to the class
     */
    void update(CUDAStream stream) { }

    /**
     * Initialize the class with the provided streams,
     * for example new streams that can be provided by {@link RetrieveNewStream#retrieve()}
     * @param streams a stream that should be associated to the class
     */
    void update(Collection<CUDAStream> streams) { }

    /**
     * Cleanup the internal state of the class, if required;
     */
    void cleanup() { }
}
