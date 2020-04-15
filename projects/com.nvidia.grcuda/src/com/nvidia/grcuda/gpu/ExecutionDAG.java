package com.nvidia.grcuda.gpu;

import com.oracle.truffle.api.interop.TruffleObject;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Directed Acyclic Graph (DAG) that represents the execution flow of GrCUDA kernels and other
 * computations. Each vertex is a computation, and an edge between vertices represents a dependency
 * such that the end vertex must wait for the start vertex to finish before starting.
 */
public class ExecutionDAG implements TruffleObject {

    private final List<DAGVertex> vertices = new ArrayList<>();
    private final List<DAGEdge> edges = new ArrayList<>();

    /**
     * Current frontier of the DAG, i.e. vertices with no children.
     */
    private List<DAGVertex> frontier = new ArrayList<>();

    /**
     * Add a new computation to the graph, and compute its dependencies.
     * @param kernel a kernel computation, containing kernel configuration and input arguments.
     */
    public void append(GrCUDAComputationalElement kernel) {
        // Add it to the list of vertices;
        DAGVertex newVertex = new DAGVertex(kernel);

        //////////////////////////////
        // Compute dependencies with other vertices in the DAG frontier, and create edges;
        //////////////////////////////

        // TODO: the frontier is not just composed of vertices without children!!
        //  I need to consider for each kernel the arguments that haven't been "covered" by a new vertex

        // For each vertex in the frontier, compute dependencies of the vertex;
        for (DAGVertex frontierVertex : frontier) {
            List<Object> dependencies = computeDependencies(frontierVertex, newVertex);
            if (dependencies.size() > 0) {
                // Create a new edge between the two vertices (book-keeping is automatic);
                new DAGEdge(frontierVertex, newVertex, dependencies);
            }
        }
        // Remove from the frontier vertices that no longer belong to it;
        frontier = frontier.stream().filter(DAGVertex::isFrontier).collect(Collectors.toList());
        // Add the new vertex to the frontier if it has no children;
        if (newVertex.isFrontier) {
            frontier.add(newVertex);
        }
    }

    private List<Object> computeDependencies(DAGVertex startVertex, DAGVertex endVertex) {
        return startVertex.getKernel().computeDependencies(endVertex.getKernel());
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
     * Simple vertex class used to encapsulate {@link KernelExecution}.
     */
    public class DAGVertex {

        private final GrCUDAComputationalElement kernel;
        private final int id;

        /**
         * False only if the vertex has parent vertices.
         */
        private boolean isStart = true;
        /**
         * False only if the vertex has child vertices.
         */
        private boolean isFrontier = true;
        /**
         * List of edges that connect this vertex to its parents (they are the start of each edge).
         */
        private final List<DAGEdge> parents = new ArrayList<>();
        /**
         * List of edges that connect this vertex to its children (they are the end of each edge).
         */
        private final List<DAGEdge> children = new ArrayList<>();

        DAGVertex(GrCUDAComputationalElement kernel) {
            this.kernel = kernel;
            this.id = getNumVertices();
            vertices.add(this);
        }

        GrCUDAComputationalElement getKernel() {
            return kernel;
        }

        int getId() {
            return id;
        }

        public boolean isStart() {
            return isStart;
        }

        public boolean isFrontier() {
            return isFrontier;
        }

        public List<DAGEdge> getParents() {
            return parents;
        }

        public List<DAGEdge> getChildren() {
            return children;
        }

        public void setStart(boolean start) {
            isStart = start;
        }

        public void setFrontier(boolean frontier) {
            isFrontier = frontier;
        }

        public void addParent(DAGEdge edge) {
            parents.add(edge);
            isStart = false;
        }

        public void addChild(DAGEdge edge) {
            children.add(edge);
            isFrontier = false;
        }

        @Override
        public String toString() {
            return "V(" +
                    ", id=" + id +
                    ", isStart=" + isStart +
                    ", isFrontier=" + isFrontier +
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
         * List of objects that represents depenencies between the two vertices;
         */
        private List<Object> dependencies;

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

        DAGEdge(DAGVertex start, DAGVertex end, List<Object> dependencies) {
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

        public List<Object> getDependencies() {
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
