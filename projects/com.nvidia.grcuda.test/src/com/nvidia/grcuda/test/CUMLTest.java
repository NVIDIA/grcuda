package com.nvidia.grcuda.test;

import com.nvidia.grcuda.gpu.executioncontext.ExecutionPolicyEnum;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class CUMLTest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {

        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {ExecutionPolicyEnum.SYNC.getName(), ExecutionPolicyEnum.ASYNC.getName()},
                {true, false},
                {'S', 'D'}
        }));
    }

    private final String policy;
    private final boolean inputPrefetch;
    private final char typeChar;

    public CUMLTest(String policy, boolean inputPrefetch, char typeChar) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
        this.typeChar = typeChar;
    }

    @Test
    public void testDbscan() {
        try (Context polyglot = Context.newBuilder().allowExperimentalOptions(true).option("grcuda.ExecutionPolicy", this.policy)
                .option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(true).build()) {
            Value cu = polyglot.eval("grcuda", "CU");
            int numRows = 100;
            int numCols = 2;
            String cudaType = (typeChar == 'D') ? "double" : "float";
            Value input = cu.invokeMember("DeviceArray", cudaType, numRows, numCols);
            Value labels = cu.invokeMember("DeviceArray", "int", numRows);
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    input.getArrayElement(i).setArrayElement(j, i / 10 + j);
                }
                labels.setArrayElement(i, 0);
            }
            double eps = 0.5;
            int minSamples = 5;
            int maxBytesPerChunk = 0;
            int verbose = 1;
            try {
                Value dbscan = polyglot.eval("grcuda", "ML::cuml" + typeChar + "pDbscanFit");
                try {
                    dbscan.execute(input, numRows, numCols, eps, minSamples, labels, maxBytesPerChunk, verbose);
                    CUBLASTest.assertOutputVectorIsCorrect(numRows, labels, (Integer i) -> i / 10, this.typeChar);
                } catch (Exception e) {
                    System.out.println("warning: failed to launch cuML, skipping test");
                }
            } catch (Exception e) {
                System.out.println("warning: cuML not enabled, skipping test");
            }
        }
    }
}
