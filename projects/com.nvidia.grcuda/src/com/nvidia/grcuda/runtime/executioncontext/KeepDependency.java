package com.nvidia.grcuda.runtime.executioncontext;

import java.util.Set;

public interface KeepDependency {
    /**
     * Determine if a vertex should really be a dependency, given a set of possible dependencies.
     * The vertex is not going to be a dependency if any of its children is included in the dependency set;
     * @param vertex a vertex we want to possibly filter, if it's an unnecessary dependency
     * @param dependentVertices a list of possible dependencies
     * @return if the vertex should be kept in the dependencies
     */
    boolean keepDependency(ExecutionDAG.DAGVertex vertex, Set<ExecutionDAG.DAGVertex> dependentVertices);
}
