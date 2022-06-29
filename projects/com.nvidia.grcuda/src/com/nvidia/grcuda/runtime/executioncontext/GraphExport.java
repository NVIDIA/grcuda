/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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

package com.nvidia.grcuda.runtime.executioncontext;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Class that receives from its constructor the ExecutionDAG and has
 * the functionality of export its graphical representation in .dot format.
 * The graph will be exported to the path inserted by the user in the option argument.
 */
public class GraphExport {

    private final List<ExecutionDAG.DAGVertex> vertices;
    private final List<ExecutionDAG.DAGEdge> edges;

    public GraphExport(ExecutionDAG dag) {
        this.vertices = dag.getVertices();
        this.edges = dag.getEdges();
    }

    /**
     * If path is valid, this method creates a .dot file with the graphical representation
     * of the scheduling DAG created during the computations.
     * @param path Destination path of the .dot file.
     */
    public void graphGenerator(String path) {
        StringBuilder output;
        List<Integer> streams = new ArrayList<>();
        List<Integer> devices = new ArrayList<>();

        for (ExecutionDAG.DAGVertex vertex : vertices){
            streams.add(vertex.getComputation().getStream().getStreamNumber());
            devices.add(vertex.getComputation().getStream().getStreamDeviceId());
        }
        streams = streams.stream().distinct().collect(Collectors.toList());
        devices = devices.stream().distinct().collect(Collectors.toList());
        int offset = streams.size();

        output = new StringBuilder("digraph G {\n" +
                "\tfontname=\"Helvetica,Arial,sans-serif\"\n" +
                "\tnode [fontname=\"Helvetica,Arial,sans-serif\"]\n" +
                "\tedge [fontname=\"Helvetica,Arial,sans-serif\"]\n" +
                "\n\n");

        for (Integer device : devices) {
            output.append("\tsubgraph cluster_").append(device).append(" {\n");

            for (Integer stream : streams) {
                if (stream < 0 ) output.append("\tsubgraph cluster_").append((stream*-1)+offset).append(" {\n").append("\t\tstyle=filled;\n").append("\t\tnode [style=filled];\n");
                else output.append("\tsubgraph cluster_").append(stream).append(" {\n").append("\t\tstyle=filled;\n").append("\t\tnode [style=filled];\n");

                for (ExecutionDAG.DAGVertex vertex : vertices) {
                    if (vertex.getComputation().getStream().getStreamNumber() == stream && vertex.getComputation().getStream().getStreamDeviceId() == device) {
                        output = new StringBuilder(output + "\"CE" + vertex.getId() + vertex.getComputation().getArgumentsThatCanCreateDependencies() + "\";\n");
                    }
                }

                output = new StringBuilder(output + "\n");
                output = new StringBuilder(output + "\t\tlabel = \"stream " + stream + "\";\n" +
                        "\t\tcolor=orange;\n" +
                        "\t}\n");
            }

            output = new StringBuilder(output + "\n");
            output = new StringBuilder(output + "\t\tlabel = \"device " + device + "\";\n" +
                    "\t\tcolor=green;\n" +
                    "\t}\n");

        }

        output = new StringBuilder(output + "\n");

        for (ExecutionDAG.DAGVertex vertex : vertices) {
            if (vertex.isStart()) {
                output = new StringBuilder(output + "start -> " + "\"CE" + vertex.getId() + vertex.getComputation().getArgumentsThatCanCreateDependencies() + "\";\n");
            }
        }

        for (ExecutionDAG.DAGEdge dependency : edges) {
            output = new StringBuilder(output + dependency.toExportGraph() + ";\n");
        }

        output = new StringBuilder(output + "\n");

        for (ExecutionDAG.DAGVertex vertex : vertices) {
            if (vertex.isFrontier()) {
                output = new StringBuilder(output + "\"CE" + vertex.getId() + vertex.getComputation().getArgumentsThatCanCreateDependencies() + "\" -> end;\n");
            }
        }

        output = new StringBuilder(output + "\tstart [shape=Mdiamond];\n" +
                "\tend [shape=Msquare];\n" +
                "}");

        path = path + ".dot";
        File graph = new File(path);
        try {
            FileWriter writer = new FileWriter(graph);
            writer.write(output.toString());
            writer.close();
            System.out.println("Execution DAG successfully exported at " + path);
        } catch (IOException e) {
            System.out.println("An error occurred while exporting the Execution DAG, please check the path");
            e.printStackTrace();
        }
    }

}
