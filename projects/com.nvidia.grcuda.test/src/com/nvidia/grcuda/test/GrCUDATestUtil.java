package com.nvidia.grcuda.test;

import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;
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
     * Return a list of {@link GrCUDATestOptionsStruct}, where each elemenent is a combination of input policy options.
     * Useful to perform tests that cover all cases;
     * @return the cross-product of all options
     */
    public static Collection<Object[]> getAllOptionCombinations() {
        Collection<Object[]> options = GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {ExecutionPolicyEnum.SYNC, ExecutionPolicyEnum.ASYNC},
                {true, false},  // InputPrefetch
                {RetrieveNewStreamPolicyEnum.FIFO, RetrieveNewStreamPolicyEnum.ALWAYS_NEW},
                {RetrieveParentStreamPolicyEnum.SAME_AS_PARENT, RetrieveParentStreamPolicyEnum.DISJOINT},
                {DependencyPolicyEnum.NO_CONST, DependencyPolicyEnum.WITH_CONST},
                {true, false},  // ForceStreamAttach
        }));
        List<Object[]> combinations = new ArrayList<>();
        options.forEach(optionArray -> {
            GrCUDATestOptionsStruct newStruct = new GrCUDATestOptionsStruct(
                    (ExecutionPolicyEnum) optionArray[0], (boolean) optionArray[1],
                    (RetrieveNewStreamPolicyEnum) optionArray[2], (RetrieveParentStreamPolicyEnum) optionArray[3],
                    (DependencyPolicyEnum) optionArray[4], (boolean) optionArray[5]);
            if (!isOptionRedundantForSync(newStruct)) {
                combinations.add(new GrCUDATestOptionsStruct[]{newStruct});
            }
        });
        // Check that the number of options is correct;
        assert(combinations.size() == (2 * 2 + 2 * 2 * 2 * 2 * 2));
        return combinations;
    }

    public static Context createContextFromOptions(GrCUDATestOptionsStruct options) {
        return buildTestContext()
                .option("grcuda.ExecutionPolicy", options.policy.getName())
                .option("grcuda.InputPrefetch", String.valueOf(options.inputPrefetch))
                .option("grcuda.RetrieveNewStreamPolicy", options.retrieveNewStreamPolicy.getName())
                .option("grcuda.RetrieveParentStreamPolicy", options.retrieveParentStreamPolicy.getName())
                .option("grcuda.DependencyPolicy", options.dependencyPolicy.getName())
                .option("grcuda.ForceStreamAttach", String.valueOf(options.forceStreamAttach))
                .build();
    }

    public static Context.Builder buildTestContext() {
        return Context.newBuilder().allowAllAccess(true).allowExperimentalOptions(true).logHandler(new TestLogHandler()).option("log.grcuda.level", "SEVERE");
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


