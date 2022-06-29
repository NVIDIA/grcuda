/*
 * Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.nvidia.grcuda.runtime.computation;

import com.nvidia.grcuda.CUDAEvent;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDALogger;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.computation.streamattach.StreamAttachArchitecturePolicy;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyComputation;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.stream.DefaultStream;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import static com.nvidia.grcuda.GrCUDALogger.COMPUTATION_LOGGER;

/**
 * Basic class that represents GrCUDA computations,
 * and is used to model data dependencies between computations;
 */
public abstract class GrCUDAComputationalElement {

    private static final TruffleLogger LOGGER = GrCUDALogger.getLogger(COMPUTATION_LOGGER);

    /**
     * This list contains the original set of input arguments that are used to compute dependencies;
     */
    protected final List<ComputationArgumentWithValue> argumentsThatCanCreateDependencies;
    /**
     * Reference to the execution context where this computation is executed;
     */
    protected final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    /**
     * Reference to the stream where this computation will be executed,
     * if possible (i.e. if the computation can be executed on a custom stream).
     * Subclasses can keep an internal reference to the stream, e.g. if it can be manually modified by the user,
     * but it is required to keep that value consistent to this one if it is modified;
     */
    private CUDAStream stream = DefaultStream.get();
    /**
     * Reference to the event associated to this computation, and recorded on the stream where this computation is executed,
     * before the computation is started. It is used in order to time the execution.
     */
    private CUDAEvent eventStart;
    /**
     * Reference to the event associated to this computation, and recorded on the stream where this computation is executed,
     * after the computation is started. It offers a precise synchronization point for children computations.
     * If the computation is not executed on a stream, the event is null;
     */
    private CUDAEvent eventStop;
    /**
     * Keep track of whether this computation has already been executed, and represents a "dead" vertex in the DAG.
     * Computations that are already executed will not be considered when computing dependencies;
     */
    private boolean computationFinished = false;
    /**
     * Keep track of whether this computation has already been started, to avoid performing the same computation multiple times;
     */
    private boolean computationStarted = false;
    /**
     * Specify if this computational element represents a computation executed on the CPU,
     * such as an array access (read or write) on an {@link com.nvidia.grcuda.runtime.array.AbstractArray}.
     * CPU computations are assumed synchronous. By default it returns false;
     */
    protected boolean isComputationDoneByCPU = false;

    private final DependencyComputation dependencyComputation;

    /**
     * True IFF {@link GrCUDAComputationalElement#executionTimeMs} has been set;
     */
    private boolean executionTimeMeasured = false;

    /**
     * Execution time in milliseconds of this computation. The execution time is available only after the end of this computation,
     * and whether the time has been measured is given by {@link GrCUDAComputationalElement#executionTimeMeasured}.
     * Whether the execution time is measurable (and measured) depends on the GrCUDAComputationalElement and on user-specified settings.
     */
    private float executionTimeMs = 0;

    /**
     * Constructor that takes an argument set initializer to build the set of arguments used in the dependency computation
     * @param grCUDAExecutionContext execution context in which this computational element will be scheduled
     * @param initializer the initializer used to build the internal set of arguments considered in the dependency computation
     */
    public GrCUDAComputationalElement(AbstractGrCUDAExecutionContext grCUDAExecutionContext, InitializeDependencyList initializer) {
        this.argumentsThatCanCreateDependencies = initializer.initialize();
        // Initialize by making a copy of the original set;
        this.grCUDAExecutionContext = grCUDAExecutionContext;
        this.dependencyComputation = grCUDAExecutionContext.getDependencyBuilder().initialize(this.argumentsThatCanCreateDependencies);
    }

    /**
     * Simplified constructor that takes a list of arguments, and consider all of them in the dependency computation
     * @param grCUDAExecutionContext execution context in which this computational element will be scheduled
     * @param args the list of arguments provided to the computation. Arguments are expected to be {@link org.graalvm.polyglot.Value}
     */
    public GrCUDAComputationalElement(AbstractGrCUDAExecutionContext grCUDAExecutionContext, List<ComputationArgumentWithValue> args) {
        this(grCUDAExecutionContext, new DefaultExecutionInitializer(args));
    }

    public List<ComputationArgumentWithValue> getArgumentsThatCanCreateDependencies() {
        return argumentsThatCanCreateDependencies;
    }

    /**
     * Store the execution time for this ComputationalElement (in milliseconds)
     * @param executionTimeMs the execution time of this ComputationalElement
     */
    public void setExecutionTime(float executionTimeMs) {
        this.executionTimeMs = executionTimeMs;
        this.executionTimeMeasured = true;
        LOGGER.fine(() -> "computation (" + this + "), execution time: " + executionTimeMs + " ms");
    }

    public float getExecutionTime() {
        if (this.executionTimeMeasured) {
            return this.executionTimeMs;
        } else {
            throw new GrCUDAException("execution time for computation " + this + " has not been measured!");
        }
    }

    /**
     * Return if this computation could lead to dependencies with future computations.
     * If not, this usually means that all of its arguments have already been superseded by other computations,
     * or that the computation didn't have any arguments to begin with;
     * @return if the computation could lead to future dependencies
     */
    public boolean hasPossibleDependencies() {
        return !this.dependencyComputation.getActiveArgumentSet().isEmpty();
    }

    /**
     * Schedule this computation for future execution by the {@link AsyncGrCUDAExecutionContext}.
     * The scheduling request is separate from the {@link GrCUDAComputationalElement} instantiation
     * as we need to ensure that the the computational element subclass has been completely instantiated;
     */
    public Object schedule() throws UnsupportedTypeException {
        return this.grCUDAExecutionContext.registerExecution(this);
    }

    /**
     * Generic interface to perform the execution of this {@link GrCUDAComputationalElement}.
     * The actual execution implementation must be added by concrete computational elements.
     * The execution request will be done by the {@link AsyncGrCUDAExecutionContext}, after this computation has been scheduled
     * using {@link GrCUDAComputationalElement#schedule()}
     */
    public abstract Object execute() throws UnsupportedTypeException;

    public CUDAStream getStream() {
        return this.stream;
    }

    public void setStream(CUDAStream stream) {
        this.stream = stream;
    }

    public boolean isComputationFinished() {
        return computationFinished;
    }

    public boolean isComputationStarted() {
        return computationStarted;
    }

    public void setComputationFinished() {
        this.computationFinished = true;
    }

    public void setComputationStarted() {
        this.computationStarted = true;
    }

    public Optional<CUDAEvent> getEventStop() {
        if (eventStop != null) {
            return Optional.of(eventStop);
        } else {
            return Optional.empty();
        }
    }

    public Optional<CUDAEvent> getEventStart() {
        if (eventStart != null) {
            return Optional.of(eventStart);
        } else {
            return Optional.empty();
        }
    }

    public void setEventStop(CUDAEvent eventStop) {
        this.eventStop = eventStop;
    }

    public void setEventStart(CUDAEvent eventStart) {
        this.eventStart = eventStart;
    }

    /**
     * Find whether this computation should be done on a user-specified {@link com.nvidia.grcuda.runtime.stream.CUDAStream};
     * If not, the stream will be provided internally using the specified execution policy. By default, return false;
     * @return if the computation is done on a custom CUDA stream;
     */
    public boolean useManuallySpecifiedStream() {
        return false;
    }

    /**
     * Some computational elements, like kernels, can be executed on different {@link CUDAStream} to provide
     * parallel asynchronous execution. Other computations, such as array reads, do not require streams, or cannot be
     * executed on streams different from the {@link DefaultStream};
     * @return if this computation can be executed on a customized stream
     */
    public boolean canUseStream() {
        return false;
    }

    // TODO: currently not supported. It is not clear what the synchronization semantic for the default stream is.
    //  It is better to just always execute computations on the default stream synchronously.
//    /**
//     * Some computational elements, like some CUDA library functions, do not expose the option to use arbitrary streams.
//     * In these cases, we still allow asynchronous execution using events etc., but the computation
//     * is always executed on the default stream.
//     * If this function returns true, {@link GrCUDAComputationalElement#canUseStream()} must also be true.
//     * Otherwise, returning true has no effect;
//     * @return if this computation must be executed on the default stream;
//     */
//    public boolean mustUseDefaultStream() { return false; }

    /**
     * Provide a way to associate input arrays allocated using managed memory to the stream
     * on which this kernel is executed. This is required by pre-Pascal GPUs to allow the CPU to access
     * managed memory belonging to arrays not used by kernels running on the GPU.
     * By default, the implementation is empty, as {@link GrCUDAComputationalElement#canUseStream} is false;
     */
    public final void associateArraysToStream() {
        grCUDAExecutionContext.getArrayStreamArchitecturePolicy().execute(this::associateArraysToStreamImpl);
    }

    /**
     * Actual implementation of {@link GrCUDAComputationalElement#associateArraysToStream()},
     * to be modified by concrete computational elements;
     */
    protected void associateArraysToStreamImpl() {}

    /**
     * Retrieve how the dependency computations are computed;
     */
    public DependencyComputation getDependencyComputation() { return dependencyComputation; }

    /**
     * Set for all the {@link com.nvidia.grcuda.runtime.array.AbstractArray} in the computation if this computation is an array access.
     * This implementation is meant for GPU computations that use streams, e.g. kernels and GPU libraries.
     * CPU computations (e.g. array accesses) should re-implement this function to track the CPU.
     * GPU computations don't use custom streams only if the are synchronized (e.g. when using the sync scheduler),
     * and there's no benefit in tracking their location.
     * Locations are updated BEFORE the start of the actual computation: if another computation is scheduled after
     * the current one, it will be scheduled assuming that the data transfer for this computation has already taken place.
     * This assumption can avoid duplicate data movements, e.g. with
     * (Xr) -> ...
     * (Xr) -> ...
     * we can avoid transferring X twice, and schedule the second kernel on the GPU where X will be already present;
     */
    public void updateLocationOfArrays() {
        for (ComputationArgumentWithValue o : this.argumentsThatCanCreateDependencies) {
            // Ignore non-array arguments. Also, don't update locations if the ComputationalElement does not use streams;
            if (o.getArgumentValue() instanceof AbstractArray && this.canUseStream()) {
                AbstractArray a = (AbstractArray) o.getArgumentValue();
                // If the argument is read-only, add the location of this ComputationalElement to the array;
                if (grCUDAExecutionContext.isConstAware() && o.isConst()) {
                    a.addArrayUpToDateLocations(this.stream.getStreamDeviceId());
                } else {
                    // Clear the list of up-to-date locations: only the current device has the updated array;
                    a.resetArrayUpToDateLocations(this.stream.getStreamDeviceId());
                }
            }
        }
    }

    /**
     * Obtain the list of input arguments for this computation that are arrays;
     * @return a list of arrays that are inputs for this computation
     */
    public List<AbstractArray> getArrayArguments(){
        // Note: "argumentsThatCanCreateDependencies" is a filter applied to the original inputs,
        // so we have no guarantees that it contains all the input arrays.
        // In practice, "argumentsThatCanCreateDependencies" is already a selection of the input arrays,
        // making the filter below unnecessary.
        // If for whatever reason we have a argumentsThatCanCreateDependencies that does not contain all the input arrays,
        // we need to store the original input list in this class as well, and apply the filter below to that list.
        return this.argumentsThatCanCreateDependencies.stream()
                .filter(ComputationArgument::isArray)
                .map(a -> (AbstractArray) a.getArgumentValue())
                .collect(Collectors.toList());
    }

    /**
     * Computes if the "other" GrCUDAComputationalElement has dependencies w.r.t. this kernel,
     * such as requiring as input a value computed by this kernel;
     * @param other kernel for which we want to check dependencies, w.r.t. this kernel
     * @return the list of arguments that the two kernels have in common
     */
    public Collection<ComputationArgumentWithValue> computeDependencies(GrCUDAComputationalElement other) {
        return this.dependencyComputation.computeDependencies(other);
    }

    /**
     * Compute and return an additional stream dependency used by this computation.
     * This function is used by {@link com.nvidia.grcuda.runtime.stream.GrCUDAStreamManager} to synchronize streams
     * that might not be directly used by this computation, but that have to be synchronized for this computation
     * to take place correctly. For example, in pre-Pascal GPUs it is required to ensure that no kernel is running if
     * the array accessed is visible to the global stream.
     * The actual invocation is wrapped by a {@link StreamAttachArchitecturePolicy},
     * as the invocation depends on the GPU architecture;
     * @return An additional stream to synchronize
     */
    public final Optional<CUDAStream> additionalStreamDependency() {
        return grCUDAExecutionContext.getArrayStreamArchitecturePolicy().execute(this::additionalStreamDependencyImpl);
    }

    /**
     * Actual implementation of {@link GrCUDAComputationalElement#additionalStreamDependency}, it can be overridden
     * by concrete computations to provide additional streams for synchronization;
     * @return An additional stream to synchronize
     */
    protected Optional<CUDAStream> additionalStreamDependencyImpl() {
        return Optional.empty();
    }

    /**
     * The default initializer will simply store all the arguments,
     * and consider each of them in the dependency computations;
     */
    private static class DefaultExecutionInitializer implements InitializeDependencyList {
        private final List<ComputationArgumentWithValue> args;

        DefaultExecutionInitializer(List<ComputationArgumentWithValue> args) {
            this.args = args;
        }

        @Override
        public List<ComputationArgumentWithValue> initialize() {
            return args;
        }
    }
}
