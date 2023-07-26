/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
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
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
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

import org.graalvm.options.OptionCategory;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionStability;

import com.nvidia.grcuda.cudalibraries.cublas.CUBLASRegistry;
import com.nvidia.grcuda.cudalibraries.cuml.CUMLRegistry;
import com.nvidia.grcuda.cudalibraries.tensorrt.TensorRTRegistry;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.oracle.truffle.api.Option;

@Option.Group(GrCUDALanguage.ID)
public final class GrCUDAOptions {

    @Option(category = OptionCategory.USER, help = "Enable cuBLAS support.", stability = OptionStability.STABLE) //
    public static final OptionKey<Boolean> CuBLASEnabled = new OptionKey<>(true);

    @Option(category = OptionCategory.USER, help = "Set the location of the cuBLAS library.", stability = OptionStability.STABLE) //
    public static final OptionKey<String> CuBLASLibrary = new OptionKey<>(CUBLASRegistry.DEFAULT_LIBRARY);

    @Option(category = OptionCategory.USER, help = "Enable cuML support.", stability = OptionStability.STABLE) //
    public static final OptionKey<Boolean> CuMLEnabled = new OptionKey<>(true);

    @Option(category = OptionCategory.USER, help = "Set the location of the cuML library.", stability = OptionStability.STABLE) //
    public static final OptionKey<String> CuMLLibrary = new OptionKey<>(CUMLRegistry.DEFAULT_LIBRARY);
  
    @Option(category = OptionCategory.USER, help = "Enable cuSPARSE support.", stability = OptionStability.STABLE) //
    public static final OptionKey<Boolean> CuSPARSEEnabled = new OptionKey<>(true);

    @Option(category = OptionCategory.USER, help = "Set the location of the cuSPARSE library.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> CuSPARSELibrary = new OptionKey<>(CUSPARSERegistry.DEFAULT_LIBRARY);

    @Option(category = OptionCategory.USER, help = "Enable TensorRT support.", stability = OptionStability.STABLE) //
    public static final OptionKey<Boolean> TensorRTEnabled = new OptionKey<>(GrCUDAOptionMap.DEFAULT_TENSORRT_ENABLED);

    @Option(category = OptionCategory.USER, help = "Set the location of the TensorRT library.", stability = OptionStability.STABLE) //
    public static final OptionKey<String> TensorRTLibrary = new OptionKey<>(TensorRTRegistry.DEFAULT_LIBRARY);

    @Option(category = OptionCategory.USER, help = "Log the execution time of GrCUDA computations using timers.", stability = OptionStability.STABLE) //
    public static final OptionKey<Boolean> EnableComputationTimers = new OptionKey<>(GrCUDAOptionMap.DEFAULT_ENABLE_COMPUTATION_TIMERS);

    @Option(category = OptionCategory.USER, help = "Choose the scheduling policy of GrCUDA computations.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> ExecutionPolicy = new OptionKey<>(GrCUDAOptionMap.DEFAULT_EXECUTION_POLICY.toString());

    @Option(category = OptionCategory.USER, help = "Choose how data dependencies between GrCUDA computations are computed.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> DependencyPolicy = new OptionKey<>(GrCUDAOptionMap.DEFAULT_DEPENDENCY_POLICY.toString());

    @Option(category = OptionCategory.USER, help = "Choose how streams for new GrCUDA computations are created.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> RetrieveNewStreamPolicy = new OptionKey<>(GrCUDAOptionMap.DEFAULT_RETRIEVE_STREAM_POLICY.toString());

    @Option(category = OptionCategory.USER, help = "Choose how streams for new GrCUDA computations are obtained from parent computations.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> RetrieveParentStreamPolicy = new OptionKey<>(GrCUDAOptionMap.DEFAULT_PARENT_STREAM_POLICY.toString());

    @Option(category = OptionCategory.USER, help = "Force the use of array stream attaching even when not required (e.g. post-Pascal GPUs).", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<Boolean> ForceStreamAttach = new OptionKey<>(GrCUDAOptionMap.DEFAULT_FORCE_STREAM_ATTACH);

    @Option(category = OptionCategory.USER, help = "Always prefetch input arrays to GPU if possible (e.g. post-Pascal GPUs).", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<Boolean> InputPrefetch = new OptionKey<>(GrCUDAOptionMap.DEFAULT_INPUT_PREFETCH);

    @Option(category = OptionCategory.USER, help = "Set how many GPUs can be used during computation. It must be at least 1, and if > 1 more than 1 GPUs are used (if available).", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<Integer> NumberOfGPUs = new OptionKey<>(GrCUDAOptionMap.DEFAULT_NUMBER_OF_GPUs);

    @Option(category = OptionCategory.USER, help = "Choose the heuristic that manages how GPU computations are mapped to devices, if multiple GPUs are available.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> DeviceSelectionPolicy = new OptionKey<>(GrCUDAOptionMap.DEFAULT_DEVICE_SELECTION_POLICY.toString());

    @Option(category = OptionCategory.USER, help = "Select a managed memory memAdvise flag, if multiple GPUs are available.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> MemAdvisePolicy = new OptionKey<>(GrCUDAOptionMap.DEFAULT_MEM_ADVISE_POLICY.toString());

    @Option(category = OptionCategory.USER, help = "Set the location of the CSV file that contains the estimated bandwidth between each CPU and GPU in the system.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> BandwidthMatrix = new OptionKey<>(GrCUDAOptionMap.DEFAULT_BANDWIDTH_MATRIX);

    @Option(category = OptionCategory.USER, help = "When selecting a device, do not give priority to devices that have less than this percentage of data already available, " +
            "in some DeviceSelectionPolicies such as min-transfer-size. A lower percentage favors exploitation, a high percentage favors exploration.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<Double> DataThreshold = new OptionKey<>(GrCUDAOptionMap.DEFAULT_DATA_THRESHOLD);

    @Option(category = OptionCategory.USER, help = "Add this option to dump scheduling DAG. Specify the destination path and the file name as value of the option (e.g. ../../../ExecutionDAG). File will be saved with .dot extension.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> ExportDAG = new OptionKey<>(GrCUDAOptionMap.DEFAULT_EXPORT_DAG);
}

