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
package com.nvidia.grcuda.runtime.executioncontext;

import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.oracle.truffle.api.interop.TruffleObject;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Directed Acyclic Graph (DAG) that represents the execution flow of GrCUDA kernels and other
 * computations. Each vertex is a computation, and an edge between vertices represents a dependency
 * such that the end vertex must wait for the start vertex to finish before starting.
 */
public class ExecutionDAG implements TruffleObject {

    private final List<DAGVertex> vertices = new ArrayList<>();
    private final List<DAGEdge> edges = new ArrayList<>();
    private final KeepDependency keepDependency;

    /**
     * Current frontier of the DAG, i.e. vertices with no children.
     */
    private List<DAGVertex> frontier = new ArrayList<>();

    public ExecutionDAG(DependencyPolicyEnum dependencyPolicy) {
        switch (dependencyPolicy) {
            case WITH_CONST:
                this.keepDependency = new WithConstKeepDependency();
                break;
            case NO_CONST:
                this.keepDependency = new DefaultKeepDependency();
                break;
            default:
                this.keepDependency = new DefaultKeepDependency();
        }
    }

    /**
     * Add a new computation to the graph, and compute its dependencies.
     * @param kernel a kernel computation, containing kernel configuration and input arguments
     * @return the new vertex that has been appended to the DAG
     */
    public DAGVertex append(GrCUDAComputationalElement kernel) {
        // Add it to the list of vertices;
        DAGVertex newVertex = new DAGVertex(kernel);

        //////////////////////////////
        // Compute dependencies with other vertices in the DAG frontier, and create edges;
        //////////////////////////////

        // For each vertex in the frontier, compute dependencies of the vertex;

        // Collect the vertices from which there are dependencies;
        Map<DAGVertex, Collection<ComputationArgumentWithValue>> dependentVerticesMap = new HashMap<>();
        List<DAGVertex> dependentVertices = new ArrayList<>();
        for (DAGVertex frontierVertex : cleanFrontier()) {
            Collection<ComputationArgumentWithValue> dependencies = computeDependencies(frontierVertex, newVertex);
            if (dependencies.size() > 0) {
                dependentVerticesMap.put(frontierVertex, dependencies);
                dependentVertices.add(frontierVertex);
            }
        }

        // Filter dependencies that are unnecessary. For example,
        //   if a computation C depends on computations A and B, and B depends on A;
        dependentVertices = dependentVertices.stream()
                .filter(v -> keepDependency.keepDependency(v, dependentVerticesMap.keySet()))
                .collect(Collectors.toList());

        // Create new edges;
        for (DAGVertex dependentVertex : dependentVertices) {
            // Create a new edge between the two vertices (book-keeping is automatic);
            new DAGEdge(dependentVertex, newVertex, dependentVerticesMap.get(dependentVertex));
        }

        // Remove from the frontier vertices that no longer belong to it;
        frontier = cleanFrontier();
        // Add the new vertex to the frontier if it has no children;
        if (newVertex.isFrontier()) {
            frontier.add(newVertex);
        }
        return newVertex;
    }

    private Collection<ComputationArgumentWithValue> computeDependencies(DAGVertex startVertex, DAGVertex endVertex) {
        return startVertex.getComputation().computeDependencies(endVertex.getComputation());
    }

    public List<DAGVertex> getVertices() {
        return vertices;
    }

    public List<DAGEdge> getEdges() {
        return edges;
    }

    public int getNumVertices() {
        return vertices.size();
    }

    public int getNumEdges() {
        return edges.size();
    }

    public List<DAGVertex> getFrontier() {
        return cleanFrontier();
    }

    /**
     * Ensure that the internal representation of the frontier is up-to-date.
     * Whether a vertex is part of the frontier can change dynamically (e.g. if a vertex computation is over),
     * and we have to ensure that the "cached" internal frontier is up-to-date every time it is accessed;
     * @return the updated DAG frontier
     */
    private List<DAGVertex> cleanFrontier() {
        frontier = frontier.stream().filter(DAGVertex::isFrontier).collect(Collectors.toList());
        return frontier;
    }

    @Override
    public String toString() {
        return "DAG(" +
                "|V|=" + vertices.size() +
                ", |E|=" + edges.size() +
                "\nvertices=\n" + vertices.stream().map(Object::toString).collect(Collectors.joining(",\n")) +
                ')';
    }

    /**
     * By default, keep all dependencies;
     */
    private static class DefaultKeepDependency implements KeepDependency {
        @Override
        public boolean keepDependency(DAGVertex vertex, Set<DAGVertex> dependentVertices) {
            return true;
        }
    }

    private static class WithConstKeepDependency implements KeepDependency {
        /**
         * Determine if a vertex should really be a dependency, given a set of possible dependencies.
         * The vertex is not going to be a dependency if any of its children is included in the dependency set;
         * @param vertex a vertex we want to possibly filter, if it's an unnecessary dependency
         * @param dependentVertices a list of possible dependencies
         * @return if the vertex should be kept in the dependencies
         */
        @Override
        public boolean keepDependency(DAGVertex vertex, Set<DAGVertex> dependentVertices) {
            // Perform a BFS starting from the children of "vertex";
            Queue<DAGVertex> queue = new ArrayDeque<>(vertex.getChildVertices());
            Set<DAGVertex> visitedVertices = new HashSet<>();

            while (!queue.isEmpty()) {
                DAGVertex currentVertex = queue.poll();
                // If the current vertex is in the set of candidate dependencies, we can filter it out;
                if (dependentVertices.contains(currentVertex)) {
                    return false;
                } else if (!visitedVertices.contains(currentVertex)) {
                    // Add children to the queue, but only if the current vertex hasn't been seen yet;
                    visitedVertices.add(currentVertex);
                    queue.addAll(currentVertex.getChildVertices());
                }
            }
            return true;
        }
    }

    /**
     * Simple vertex class used to encapsulate {@link GrCUDAComputationalElement}.
     */
    public class DAGVertex {

        private final GrCUDAComputationalElement computation;
        private final int id;

        /**
         * False only if the vertex has parent vertices.
         */
        private boolean isStart = true;
        /**
         * List of edges that connect this vertex to its parents (they are the start of each edge).
         */
        private final List<DAGEdge> parents = new ArrayList<>();
        /**
         * List of edges that connect this vertex to its children (they are the end of each edge).
         */
        private final List<DAGEdge> children = new ArrayList<>();

        DAGVertex(GrCUDAComputationalElement computation) {
            this.computation = computation;
            this.id = getNumVertices();
            vertices.add(this);
        }

        public GrCUDAComputationalElement getComputation() {
            return computation;
        }

        int getId() {
            return id;
        }

        public boolean isStart() {
            return isStart;
        }

        /**
         * A vertex is considered part of the DAG frontier if it could lead to dependencies.
         * In general, a vertex is not part of the frontier only if it has no arguments, it has already been executed,
         * or all its arguments have already been superseded by the arguments of computations that depends on this one;
         * @return if this vertex is part of the DAG frontier
         */
        public boolean isFrontier() {
            return computation.hasPossibleDependencies() && !computation.isComputationFinished();
        }

        /**
         * Check if this vertex corresponds to a computation that can be immediately executed.
         * This usually happens if the computations has no parents, or all the parents have already completed their execution;
         * @return if the computation can be started immediately
         */
        public boolean isExecutable() {
            return !computation.isComputationStarted() && (parents.isEmpty() || allParentsHaveFinishedComputation());
        }

        private boolean allParentsHaveFinishedComputation() {
            for (DAGEdge e : parents) {
                if (!e.getStart().getComputation().isComputationFinished()) return false;
            }
            return true;
        }

        public List<DAGEdge> getParents() {
            return parents;
        }

        public List<DAGEdge> getChildren() {
            return children;
        }

        public List<DAGVertex> getParentVertices() { return parents.stream().map(DAGEdge::getStart).collect(Collectors.toList()); }

        public List<DAGVertex> getChildVertices() { return children.stream().map(DAGEdge::getEnd).collect(Collectors.toList()); }

        public List<GrCUDAComputationalElement> getParentComputations() {
            return parents.stream().map(e -> e.getStart().getComputation()).collect(Collectors.toList());
        }

        public List<GrCUDAComputationalElement> getChildComputations() {
            return children.stream().map(e -> e.getEnd().getComputation()).collect(Collectors.toList());
        }

        public void setStart(boolean start) {
            isStart = start;
        }

        public void addParent(DAGEdge edge) {
            parents.add(edge);
            isStart = false;
        }

        public void addChild(DAGEdge edge) {
            children.add(edge);
        }

        @Override
        public String toString() {
            return "V(" +
                    "id=" + id +
                    ", isStart=" + isStart +
                    ", isFrontier=" + this.isFrontier() +
                    ", parents=" + parents +
                    ", children=" + children +
                    ')';
        }
    }

    /**
     * Simple edge class used to connect {@link DAGVertex} with dependencies.
     * An edge from a source to a destination means that the destination computation must wait
     * for the start computation to finish before starting.
     */
    public class DAGEdge {

        final private DAGVertex start;
        final private DAGVertex end;
        final private int id;
        /**
         * Set of objects that represents depenencies between the two vertices;
         */
        private Collection<ComputationArgumentWithValue> dependencies;

        DAGEdge(DAGVertex start, DAGVertex end) {
            this.start = start;
            this.end = end;
            this.id = getNumEdges();

            // Update parents and children of the two vertices;
            start.addChild(this);
            end.addParent(this);
            // Book-keeping of the edge;
            edges.add(this);
        }

        DAGEdge(DAGVertex start, DAGVertex end, Collection<ComputationArgumentWithValue> dependencies) {
            this(start, end);
            this.dependencies = dependencies;
        }

        public DAGVertex getStart() {
            return start;
        }

        public DAGVertex getEnd() {
            return end;
        }

        public int getId() {
            return id;
        }

        public Collection<ComputationArgumentWithValue> getDependencies() {
            return dependencies;
        }

        @Override
        public String toString() {
            return "E(" +
                    "start=" + start.getId() +
                    ", end=" + end.getId() +
                    ')';
        }
    }
}
