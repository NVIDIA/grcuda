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
package com.nvidia.grcuda.test.util;

import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicy;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import org.graalvm.polyglot.Context;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class GrCUDATestUtil {
    public static Collection<Object[]> crossProduct(List<Object[]> sets) {
        int solutions = 1;
        List<Object[]> combinations = new ArrayList<>();
        for (Object[] objects : sets) {
            solutions *= objects.length;
        }
        for(int i = 0; i < solutions; i++) {
            int j = 1;
            List<Object> current = new ArrayList<>();
            for(Object[] set : sets) {
                current.add(set[(i / j) % set.length]);
                j *= set.length;
            }
            combinations.add(current.toArray(new Object[0]));
        }
        return combinations;
    }

    /**
     * Return a list of {@link GrCUDATestOptionsStruct}, where each element is a combination of input policy options.
     * Useful to perform tests that cover all cases;
     * @return the cross-product of all options
     */
    public static Collection<Object[]> getAllOptionCombinationsSingleGPU() {
        Collection<Object[]> options = GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {ExecutionPolicyEnum.SYNC, ExecutionPolicyEnum.ASYNC},
                {true, false},  // InputPrefetch
                {RetrieveNewStreamPolicyEnum.REUSE, RetrieveNewStreamPolicyEnum.ALWAYS_NEW},
                {RetrieveParentStreamPolicyEnum.SAME_AS_PARENT, RetrieveParentStreamPolicyEnum.DISJOINT},
                {DependencyPolicyEnum.NO_CONST, DependencyPolicyEnum.WITH_CONST},
                {DeviceSelectionPolicyEnum.SINGLE_GPU},
                {true, false},  // ForceStreamAttach
                {true, false},  // With and without timing of kernels
                {1},            // Number of GPUs
        }));
        List<Object[]> combinations = new ArrayList<>();
        options.forEach(optionArray -> {
            GrCUDATestOptionsStruct newStruct = new GrCUDATestOptionsStruct(
                    (ExecutionPolicyEnum) optionArray[0], (boolean) optionArray[1],
                    (RetrieveNewStreamPolicyEnum) optionArray[2], (RetrieveParentStreamPolicyEnum) optionArray[3],
                    (DependencyPolicyEnum) optionArray[4], (DeviceSelectionPolicyEnum) optionArray[5],
                    (boolean) optionArray[6], (boolean) optionArray[7], (int) optionArray[8]);
            if (!isOptionRedundantForSync(newStruct)) {
                combinations.add(new GrCUDATestOptionsStruct[]{newStruct});
            }
        });
        // Check that the number of options is correct <(sync + async) * logging>;
        assert(combinations.size() == (2 * 2 + 2 * 2 * 2 * 2 * 2) * 2);
        return combinations;
    }

    /**
     * Return a list of {@link GrCUDATestOptionsStruct}, where each element is a combination of input policy options.
     * Cover testing options for multi-GPU systems. Do not consider the sync scheduling as it does not support multiple GPUs;
     * @return the cross-product of all options
     */
    public static Collection<Object[]> getAllOptionCombinationsMultiGPU() {
        Collection<Object[]> options = GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {ExecutionPolicyEnum.ASYNC},
                {true, false},  // InputPrefetch
                {RetrieveNewStreamPolicyEnum.REUSE, RetrieveNewStreamPolicyEnum.ALWAYS_NEW}, // Simplify number of tests, don't use all options;
                {RetrieveParentStreamPolicyEnum.SAME_AS_PARENT, RetrieveParentStreamPolicyEnum.DISJOINT, RetrieveParentStreamPolicyEnum.MULTIGPU_EARLY_DISJOINT, RetrieveParentStreamPolicyEnum.MULTIGPU_DISJOINT},
                {DependencyPolicyEnum.WITH_CONST, DependencyPolicyEnum.NO_CONST},   // Simplify number of tests, don't use all options;
                {DeviceSelectionPolicyEnum.SINGLE_GPU, DeviceSelectionPolicyEnum.STREAM_AWARE, DeviceSelectionPolicyEnum.ROUND_ROBIN,
                        DeviceSelectionPolicyEnum.MIN_TRANSFER_SIZE, DeviceSelectionPolicyEnum.MINMIN_TRANSFER_TIME, DeviceSelectionPolicyEnum.MINMAX_TRANSFER_TIME},
                {false, true},  // ForceStreamAttach, simplify number of tests, don't use all options;
                {true, false},  // With and without timing of kernels
                {2, 4, 8},  // Number of GPUs
        }));
        List<Object[]> combinations = new ArrayList<>();
        options.forEach(optionArray -> {
            GrCUDATestOptionsStruct newStruct = new GrCUDATestOptionsStruct(
                    (ExecutionPolicyEnum) optionArray[0], (boolean) optionArray[1],
                    (RetrieveNewStreamPolicyEnum) optionArray[2], (RetrieveParentStreamPolicyEnum) optionArray[3],
                    (DependencyPolicyEnum) optionArray[4], (DeviceSelectionPolicyEnum) optionArray[5],
                    (boolean) optionArray[6], (boolean) optionArray[7], (int) optionArray[8]);
            combinations.add(new GrCUDATestOptionsStruct[]{newStruct});
        });
        // Check that the number of options is correct;
        assert(combinations.size() == (2 * 2 * 4 * 2 * 6 * 2 * 2 * 3));
        return combinations;
    }

    public static Context createContextFromOptions(GrCUDATestOptionsStruct options, int numberOfGPUs) {
        return buildTestContext()
                .option("grcuda.ExecutionPolicy", options.policy.toString())
                .option("grcuda.InputPrefetch", String.valueOf(options.inputPrefetch))
                .option("grcuda.RetrieveNewStreamPolicy", options.retrieveNewStreamPolicy.toString())
                .option("grcuda.RetrieveParentStreamPolicy", options.retrieveParentStreamPolicy.toString())
                .option("grcuda.DependencyPolicy", options.dependencyPolicy.toString())
                .option("grcuda.DeviceSelectionPolicy", options.deviceSelectionPolicy.toString())
                .option("grcuda.ForceStreamAttach", String.valueOf(options.forceStreamAttach))
                .option("grcuda.EnableComputationTimers", String.valueOf(options.timeComputation))
                .option("grcuda.NumberOfGPUs", String.valueOf(numberOfGPUs))
                .build();
    }

    public static Context createContextFromOptions(GrCUDATestOptionsStruct options) {
        return GrCUDATestUtil.createContextFromOptions(options, options.numberOfGPUs);
    }

    public static Context.Builder buildTestContext() {
        return Context.newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .logHandler(new TestLogHandler())
                .option("log.grcuda.com.nvidia.grcuda.level", "WARNING")
//                .option("log.grcuda." + GrCUDALogger.COMPUTATION_LOGGER + ".level", "FINE")  // Uncomment to print kernel log;
                ;
    }

    /**
     * If the execution policy is "sync", we don't need to test all combinations of flags that are specific to
     * the async scheduler. So we can simply keep the default values for them (as they are unused anyway)
     * and flag all other combinations as redundant;
     * @param options a combination of input options for GrCUDA
     * @return if the option combination is redundant for the sync scheduler
     */
    private static boolean isOptionRedundantForSync(GrCUDATestOptionsStruct options) {
        if (options.policy.equals(ExecutionPolicyEnum.SYNC)) {
            return options.retrieveNewStreamPolicy.equals(RetrieveNewStreamPolicyEnum.ALWAYS_NEW) ||
                    options.retrieveParentStreamPolicy.equals(RetrieveParentStreamPolicyEnum.DISJOINT) ||
                    options.dependencyPolicy.equals(DependencyPolicyEnum.WITH_CONST);
        }
        return false;
    }
}


