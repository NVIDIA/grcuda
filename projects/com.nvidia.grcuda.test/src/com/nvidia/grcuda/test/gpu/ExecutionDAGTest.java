package com.nvidia.grcuda.test.gpu;

import com.nvidia.grcuda.gpu.ConfiguredKernel;
import com.nvidia.grcuda.gpu.ExecutionDAG;
import com.nvidia.grcuda.gpu.KernelExecution;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class ExecutionDAGTest {

    private static class KernelExecutionTest extends KernelExecution {

        List<Integer> testArguments;

        KernelExecutionTest(List<Integer> arguments) {
            super(new ConfiguredKernel(null, null), null);
            this.testArguments = arguments;
        }

        @Override
        public boolean hasDependency(KernelExecution other) {
            return true;
        }
    }

    @Test
    public void executionDAGConstructorTest() {
        ExecutionDAG dag = new ExecutionDAG();
        assertTrue(dag.getVertices().isEmpty());
        assertTrue(dag.getEdges().isEmpty());
        assertTrue(dag.getFrontier().isEmpty());
        assertEquals(0, dag.getNumVertices());
        assertEquals(0, dag.getNumEdges());
    }

    @Test
    public void addVertexToDAGTest() {
        ExecutionDAG dag = new ExecutionDAG();
        // Create two mock kernel executions;
        KernelExecutionTest kernel1 = new KernelExecutionTest(Arrays.asList(1, 2, 3));
        KernelExecutionTest kernel2 = new KernelExecutionTest(Arrays.asList(1, 2, 3));

        dag.append(kernel1);

        assertEquals(1, dag.getNumVertices());
        assertEquals(0, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
        assertTrue(dag.getFrontier().get(0).isFrontier());
        assertTrue(dag.getFrontier().get(0).isStart());

        dag.append(kernel2);

        assertEquals(2, dag.getNumVertices());
        assertEquals(1, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(1, dag.getFrontier().size());
        assertEquals(dag.getVertices().get(1), dag.getFrontier().get(0));
        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(1).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isStart());
        // Check if the first vertex is a parent of the second;
        assertEquals(dag.getVertices().get(0), dag.getVertices().get(1).getParents().get(0).getStart());
        // Check if the second vertex is a child of the first;
        assertEquals(dag.getVertices().get(1), dag.getVertices().get(0).getChildren().get(0).getEnd());
    }
}
