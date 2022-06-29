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
package com.nvidia.grcuda;

import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.computation.memadvise.MemAdviserEnum;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.StopIterationException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownKeyException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionValues;

import java.io.File;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;

@ExportLibrary(InteropLibrary.class)
public class GrCUDAOptionMap implements TruffleObject {

    /**
     * Store options using the option name and its value;
     */
    private final HashMap<String, Object> optionsMap;
    /**
     * Store a mapping between GrCUDA's Truffle options and their names, as strings.
     * OptionKeys are assumed to be immutable, so this map must be read-only as well;
     */
    private final HashMap<OptionKey<?>, String> optionNames;

    private static final TruffleLogger LOGGER = TruffleLogger.getLogger(GrCUDALanguage.ID, "com.nvidia.grcuda.GrCUDAOptionMap");

    public static final ExecutionPolicyEnum DEFAULT_EXECUTION_POLICY = ExecutionPolicyEnum.ASYNC;
    public static final DependencyPolicyEnum DEFAULT_DEPENDENCY_POLICY = DependencyPolicyEnum.NO_CONST;
    public static final RetrieveNewStreamPolicyEnum DEFAULT_RETRIEVE_STREAM_POLICY = RetrieveNewStreamPolicyEnum.REUSE;
    public static final RetrieveParentStreamPolicyEnum DEFAULT_PARENT_STREAM_POLICY = RetrieveParentStreamPolicyEnum.SAME_AS_PARENT;
    public static final DeviceSelectionPolicyEnum DEFAULT_DEVICE_SELECTION_POLICY = DeviceSelectionPolicyEnum.SINGLE_GPU;
    public static final MemAdviserEnum DEFAULT_MEM_ADVISE_POLICY = MemAdviserEnum.NONE;
    public static final boolean DEFAULT_INPUT_PREFETCH = false;  // Value obtained from the input flags;
    public static final boolean DEFAULT_FORCE_STREAM_ATTACH = false;
    public static final boolean DEFAULT_TENSORRT_ENABLED = false;
    public static final boolean DEFAULT_ENABLE_COMPUTATION_TIMERS = false;
    public static final Integer DEFAULT_NUMBER_OF_GPUs = 1;
    public static final String DEFAULT_BANDWIDTH_MATRIX = System.getenv("GRCUDA_HOME") + File.separatorChar +
            "projects" + File.separatorChar + "resources" + File.separatorChar +
            "connection_graph" + File.separatorChar + "datasets" + File.separatorChar + "connection_graph.csv";
    public static final double DEFAULT_DATA_THRESHOLD = 0.1;
    public static final String DEFAULT_EXPORT_DAG = "false";

    public GrCUDAOptionMap(OptionValues options) {
        optionsMap = new HashMap<>();
        optionNames = new HashMap<>();

        // Store the name and value of each option;
        // Map each OptionKey to its name, to retrieve values inside GrCUDA;
        options.getDescriptors().forEach(o -> {
            optionsMap.put(o.getName(), options.get(o.getKey()));
            optionNames.put(o.getKey(), o.getName());
        });

        // Parse individual options;

        // Stream retrieval policy;
        optionsMap.replace(optionNames.get(GrCUDAOptions.RetrieveNewStreamPolicy), parseRetrieveStreamPolicy(options.get(GrCUDAOptions.RetrieveNewStreamPolicy)));
        // How streams are obtained from parent computations;
        optionsMap.replace(optionNames.get(GrCUDAOptions.RetrieveParentStreamPolicy), parseParentStreamPolicy(options.get(GrCUDAOptions.RetrieveParentStreamPolicy)));
        // Dependency computation policy;
        optionsMap.replace(optionNames.get(GrCUDAOptions.DependencyPolicy), parseDependencyPolicy(options.get(GrCUDAOptions.DependencyPolicy)));
        // Execution policy;
        optionsMap.replace(optionNames.get(GrCUDAOptions.ExecutionPolicy), parseExecutionPolicy(options.get(GrCUDAOptions.ExecutionPolicy)));
        // Device selection policy;
        optionsMap.replace(optionNames.get(GrCUDAOptions.DeviceSelectionPolicy), parseDeviceSelectionPolicy(options.get(GrCUDAOptions.DeviceSelectionPolicy)));
        // Memory advise policy;
        optionsMap.replace(optionNames.get(GrCUDAOptions.MemAdvisePolicy), parseMemAdvisePolicy(options.get(GrCUDAOptions.MemAdvisePolicy)));
    }

    /**
     * Obtain the option value starting from the OptionKey;
     */
    private Object getOptionValueFromOptionKey(OptionKey<?> optionKey) {
        return optionsMap.get(optionNames.get(optionKey));
    }

    // Enforces immutability;
    public HashMap<String, Object> getOptions(){
        return new HashMap<>(optionsMap);
    }

    private static ExecutionPolicyEnum parseExecutionPolicy(String policyString) {
        if (policyString.equals(ExecutionPolicyEnum.SYNC.toString())) return ExecutionPolicyEnum.SYNC;
        else if (policyString.equals(ExecutionPolicyEnum.ASYNC.toString())) return ExecutionPolicyEnum.ASYNC;
        else {
            LOGGER.severe("unknown execution policy=" + policyString + "; using default=" + DEFAULT_EXECUTION_POLICY);
            return DEFAULT_EXECUTION_POLICY;
        }
    }

    private static DependencyPolicyEnum parseDependencyPolicy(String policyString) {
        if (policyString.equals(DependencyPolicyEnum.WITH_CONST.toString())) return DependencyPolicyEnum.WITH_CONST;
        else if (policyString.equals(DependencyPolicyEnum.NO_CONST.toString())) return DependencyPolicyEnum.NO_CONST;
        else {
            LOGGER.warning("Warning: unknown dependency policy=" + policyString + "; using default=" + DEFAULT_DEPENDENCY_POLICY);
            return DEFAULT_DEPENDENCY_POLICY;
        }
    }

    private static RetrieveNewStreamPolicyEnum parseRetrieveStreamPolicy(String policyString) {
        if (policyString.equals(RetrieveNewStreamPolicyEnum.REUSE.toString())) return RetrieveNewStreamPolicyEnum.REUSE;
        else if (policyString.equals(RetrieveNewStreamPolicyEnum.ALWAYS_NEW.toString())) return RetrieveNewStreamPolicyEnum.ALWAYS_NEW;
        else {
            LOGGER.warning("Warning: unknown new stream retrieval policy=" + policyString + "; using default=" + DEFAULT_RETRIEVE_STREAM_POLICY);
            return DEFAULT_RETRIEVE_STREAM_POLICY;
        }
    }

    private static RetrieveParentStreamPolicyEnum parseParentStreamPolicy(String policyString) {
        if (Objects.equals(policyString, RetrieveParentStreamPolicyEnum.DISJOINT.toString())) return RetrieveParentStreamPolicyEnum.DISJOINT;
        else if (Objects.equals(policyString, RetrieveParentStreamPolicyEnum.SAME_AS_PARENT.toString())) return RetrieveParentStreamPolicyEnum.SAME_AS_PARENT;
        else if (Objects.equals(policyString, RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT.toString())) return RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT;
        else if (Objects.equals(policyString, RetrieveParentStreamPolicyEnum.MULTIGPU_DISJOINT.toString())) return RetrieveParentStreamPolicyEnum.MULTIGPU_DISJOINT;
        else {
            LOGGER.warning("Warning: unknown parent stream retrieval policy=" + policyString + "; using default=" + DEFAULT_PARENT_STREAM_POLICY);
            return DEFAULT_PARENT_STREAM_POLICY;
        }
    }

    private static DeviceSelectionPolicyEnum parseDeviceSelectionPolicy(String policyString) {
        if (Objects.equals(policyString, DeviceSelectionPolicyEnum.SINGLE_GPU.toString())) return DeviceSelectionPolicyEnum.SINGLE_GPU;
        else if (Objects.equals(policyString, DeviceSelectionPolicyEnum.ROUND_ROBIN.toString())) return DeviceSelectionPolicyEnum.ROUND_ROBIN;
        else if (Objects.equals(policyString, DeviceSelectionPolicyEnum.STREAM_AWARE.toString())) return DeviceSelectionPolicyEnum.STREAM_AWARE;
        else if (Objects.equals(policyString, DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE.toString())) return DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE;
        else if (Objects.equals(policyString, DeviceSelectionPolicyEnum.MINMIN_TRANSFER_TIME.toString())) return DeviceSelectionPolicyEnum.MINMIN_TRANSFER_TIME;
        else if (Objects.equals(policyString, DeviceSelectionPolicyEnum.MINMAX_TRANSFER_TIME.toString())) return DeviceSelectionPolicyEnum.MINMAX_TRANSFER_TIME;
        else {
            LOGGER.warning("Warning: unknown device selection policy=" + policyString + "; using default=" + DEFAULT_DEVICE_SELECTION_POLICY);
            return DEFAULT_DEVICE_SELECTION_POLICY;
        }
    }

    private static MemAdviserEnum parseMemAdvisePolicy(String policyString) {
        if (Objects.equals(policyString, MemAdviserEnum.ADVISE_READ_MOSTLY.toString())) return MemAdviserEnum.ADVISE_READ_MOSTLY;
        else if (Objects.equals(policyString, MemAdviserEnum.ADVISE_PREFERRED_LOCATION.toString())) return MemAdviserEnum.ADVISE_PREFERRED_LOCATION;
        else if (Objects.equals(policyString, MemAdviserEnum.NONE.toString())) return MemAdviserEnum.NONE;
        else {
            LOGGER.warning("Warning: unknown memory advice policy=" + policyString + "; using default=" + DEFAULT_MEM_ADVISE_POLICY);
            return DEFAULT_MEM_ADVISE_POLICY;
        }
    }

    public Boolean isCuBLASEnabled(){
        return (Boolean) getOptionValueFromOptionKey(GrCUDAOptions.CuBLASEnabled);
    }

    public String getCuBLASLibrary(){
        return (String) getOptionValueFromOptionKey(GrCUDAOptions.CuBLASLibrary);
    }

    public Boolean isCuMLEnabled(){
        return (Boolean) getOptionValueFromOptionKey(GrCUDAOptions.CuMLEnabled);
    }

    public String getCuMLLibrary(){
        return (String) getOptionValueFromOptionKey(GrCUDAOptions.CuMLLibrary);
    }

    public Boolean isCuSPARSEEnabled(){
        return (Boolean) getOptionValueFromOptionKey(GrCUDAOptions.CuSPARSEEnabled);
    }

    public String getCuSPARSELibrary(){
        return (String) getOptionValueFromOptionKey(GrCUDAOptions.CuSPARSELibrary);
    }

    public ExecutionPolicyEnum getExecutionPolicy(){
        return (ExecutionPolicyEnum) getOptionValueFromOptionKey(GrCUDAOptions.ExecutionPolicy);
    }

    public DependencyPolicyEnum getDependencyPolicy(){
        return (DependencyPolicyEnum) getOptionValueFromOptionKey(GrCUDAOptions.DependencyPolicy);
    }

    public RetrieveNewStreamPolicyEnum getRetrieveNewStreamPolicy(){
        return (RetrieveNewStreamPolicyEnum) getOptionValueFromOptionKey(GrCUDAOptions.RetrieveNewStreamPolicy);
    }

    public RetrieveParentStreamPolicyEnum getRetrieveParentStreamPolicy(){
        return (RetrieveParentStreamPolicyEnum) getOptionValueFromOptionKey(GrCUDAOptions.RetrieveParentStreamPolicy);
    }

    public Boolean isForceStreamAttach(){
        return (Boolean) getOptionValueFromOptionKey(GrCUDAOptions.ForceStreamAttach);
    }

    public Boolean isInputPrefetch(){
        return (Boolean) getOptionValueFromOptionKey(GrCUDAOptions.InputPrefetch);
    }
    
    public Boolean isTimeComputation() { return (Boolean) getOptionValueFromOptionKey(GrCUDAOptions.EnableComputationTimers); }

    public Boolean isTensorRTEnabled(){
        return (Boolean) getOptionValueFromOptionKey(GrCUDAOptions.TensorRTEnabled);
    }

    public String getTensorRTLibrary(){
        return (String) getOptionValueFromOptionKey(GrCUDAOptions.TensorRTLibrary);
    }

    public Integer getNumberOfGPUs() {
        return (Integer) getOptionValueFromOptionKey(GrCUDAOptions.NumberOfGPUs);
    }

    public String getBandwidthMatrix() { return (String) getOptionValueFromOptionKey(GrCUDAOptions.BandwidthMatrix); }

    public Double getDataThreshold() {
        return (Double) getOptionValueFromOptionKey(GrCUDAOptions.DataThreshold);
    }

    public String getExportDAGPath() {
        return (String) getOptionValueFromOptionKey(GrCUDAOptions.ExportDAG);
    }

    public MemAdviserEnum getMemAdvisePolicy() {
        return (MemAdviserEnum) getOptionValueFromOptionKey(GrCUDAOptions.MemAdvisePolicy);
    }

    public DeviceSelectionPolicyEnum getDeviceSelectionPolicy() {
        return (DeviceSelectionPolicyEnum) getOptionValueFromOptionKey(GrCUDAOptions.DeviceSelectionPolicy);
    }

    public void setNumberOfGPUs(int numberOfGPUs) {
        LOGGER.info("updated the number of GPUs to use from " + getNumberOfGPUs() + " to " + numberOfGPUs);
        optionsMap.replace(optionNames.get(GrCUDAOptions.NumberOfGPUs), numberOfGPUs);
    }

    // Implement InteropLibrary;

    @ExportMessage
    public final boolean hasHashEntries(){
        return true;
    }

    @ExportMessage
    public final Object readHashValue(Object key) throws UnknownKeyException, UnsupportedMessageException {
        Object value;
        if (key instanceof String){
            value = this.optionsMap.get(key);
        }
        else {
            throw UnsupportedMessageException.create();
        }
        if (value == null) throw UnknownKeyException.create(key);
        return value.toString();
    }

    @ExportMessage
    public final long getHashSize(){
        return optionsMap.size();
    }

    @ExportMessage
    public final boolean isHashEntryReadable(Object key) {
        return key instanceof String && this.optionsMap.containsKey(key);
    }

    @ExportMessage
    public Object getHashEntriesIterator() {
        return new EntriesIterator(optionsMap.entrySet().iterator());
    }

    @ExportLibrary(InteropLibrary.class)
    public static final class EntriesIterator implements TruffleObject {
        private final Iterator<Map.Entry<String, Object>> iterator;

        private EntriesIterator(Iterator<Map.Entry<String, Object>> iterator) {
            this.iterator = iterator;
        }

        @SuppressWarnings("static-method")
        @ExportMessage
        public boolean isIterator() {
            return true;
        }

        @ExportMessage
        public boolean hasIteratorNextElement() {
            try {
                return iterator.hasNext();
            } catch(NoSuchElementException e) {
                return false;
            }
        }

        @ExportMessage
        public GrCUDAOptionTuple getIteratorNextElement() throws StopIterationException {
            if (hasIteratorNextElement()) {
                Map.Entry<String,Object> entry = iterator.next();
                return new GrCUDAOptionTuple(entry.getKey(), entry.getValue().toString());
            } else {
                throw StopIterationException.create();
            }
        }
    }

    @ExportLibrary(InteropLibrary.class)
    public static class GrCUDAOptionTuple implements TruffleObject {

        private final int SIZE = 2;
        private final String[] entry = new String[SIZE];

        public GrCUDAOptionTuple(String key, String value) {
            entry[0] = key;
            entry[1] = value;
        }

        @ExportMessage
        static boolean hasArrayElements(GrCUDAOptionTuple tuple) {
            return true;
        }

        @ExportMessage
        public final boolean isArrayElementReadable(long index) {
            return index == 0 || index == 1;
        }

        @ExportMessage
        public final Object readArrayElement(long index) throws InvalidArrayIndexException {
            if (index == 0 || index == 1) {
                return entry[(int)index];
            }
            else {
                throw InvalidArrayIndexException.create(index);
            }
        }

        @ExportMessage
        public final long getArraySize() {
            return SIZE;
        }
    }

}
