/*
 * Copyright (c) 2022 NECSTLab, Politecnico di Milano. All rights reserved.
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

package it.necst.grcuda.benchmark;


import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;

/**
 * This class stores all the results coming from a benchmark.
 * It is mainly composed of a linked list containing various BenchmarkRecord, those records are reporting information on the single phases in the benchmark (like timings etc).
 */
public class BenchmarkResults {
    public final BenchmarkConfig config;
    public LinkedList<Iteration> iterations = new LinkedList<>();
    public ArrayList<String> filteredPhases = new ArrayList<>();
    public double cpu_result;

    BenchmarkResults(BenchmarkConfig config){
        this.config = config;
        filteredPhases.add("alloc");
        filteredPhases.add("reset");
        filteredPhases.add("init");
    }

    public void startNewIteration(int iter_num, boolean time_phases){
        iterations.add(new Iteration(iter_num, time_phases));
    }
    public void addPhaseToCurrentIteration(String phaseName, double execTime){
        iterations.getLast().addPhase(phaseName, execTime);
    }
    public void saveToJsonFile() {
        try {
            ObjectMapper objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
            if(config.debug)
                System.out.println(objectMapper.writeValueAsString(this));

            String sb =
                    config.benchmarkName +
                            "_" + config.size +
                            "_" + config.numGpus +
                            "_" + config.numBlocks +
                            "_" + config.executionPolicy +
                            "_" + config.dependencyPolicy +
                            "_" + config.retrieveNewStreamPolicy +
                            "_" + config.retrieveParentStreamPolicy +
                            "_" + config.deviceSelectionPolicy +
                            "_" + config.memAdvisePolicy +
                            "_" + config.inputPrefetch +
                            "_" + config.forceStreamAttach +
                            ".json";

            objectMapper.writeValue(new File(config.results_path+"/"+ sb), this);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }


    public void setCurrentGpuResult(double gpuResult){
        iterations.getLast().gpu_result = gpuResult;
    }
    public void setCurrentCpuResult(double cpuResult){
        this.cpu_result = cpuResult;
    }
    public void setCurrentComputationSec(double computationSec){iterations.getLast().computation_sec = computationSec;}
    public void setCurrentOverheadSec(double overheadSec){iterations.getLast().overhead_sec = overheadSec;}
    public void setCurrentTotalTime(double totalTime){
        iterations.getLast().total_time_sec = totalTime;
        double tot_time_phases = 0;
        for(Phase phase : iterations.getLast().phases){
            if(!filteredPhases.contains(phase.phaseName))
                tot_time_phases += phase.executionTime_sec;
        }
        iterations.getLast().overhead_sec = totalTime-tot_time_phases;
        iterations.getLast().computation_sum_phases_sec = tot_time_phases;
    }


    public double currentGpuResult(){
        return iterations.getLast().gpu_result;
    }
    public double currentCpuResult(){
        return this.cpu_result;
    }
    public Iteration currentIteration(){return iterations.getLast();}

}

class Iteration{
    public int iteration;
    public boolean time_phases;
    public double gpu_result;

    public double computation_sec;
    public double total_time_sec;
    public double overhead_sec;
    public double computation_sum_phases_sec;

    public ArrayList<Phase> phases = new ArrayList<>();

    public Iteration(int iteration, boolean time_phases) {
        this.iteration = iteration;
        this.time_phases = time_phases;
    }

    public void addPhase(String phaseName, double execTime){phases.add(new Phase(phaseName, execTime));}

}

class Phase {
    public String phaseName;         // the phase of the benchmark that this class is representing
    public double executionTime_sec;       // the execution time of the current phase

    public Phase(String phaseName, double executionTime_sec) {
        this.phaseName = phaseName;
        this.executionTime_sec = executionTime_sec;
    }
}