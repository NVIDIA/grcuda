# Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:10:07 2020

@author: alberto.parravicini
"""


import pandas as pd
import json
import os
import numpy as np
from compute_transfer_computation_overlap import get_overlap, get_total_segment_set_length

import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy.stats.mstats import gmean
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.lines as lines
from plot_utils import COLORS, get_exp_label, get_ci_size, save_plot


DEFAULT_RES_DIR = "../../../../grcuda-data/results/scheduling_nvprof_log"

INPUT_DATE = "2020_10_11_1660_2"
OUTPUT_DATE = "2020_10_11"
PLOT_DIR = "../../../../grcuda-data/plots"

BENCHMARK_NAMES = {
    "b1": "VEC",
    "b5": "B&S", 
    "b8": "IMG",
    "b6": "ML",
    "b7": "HITS", 
    "b10": "DL"
    }
# ASYNC_POLICY_NAME = "async"   # If parsing new results;
ASYNC_POLICY_NAME = "default"  # If parsing older results;
POLICIES = ["sync", ASYNC_POLICY_NAME]
POLICIES_DICT = {ASYNC_POLICY_NAME: "Parallel Scheduler", "sync": "Serial Scheduler"}

NVPROF_HEADER_NOMETRIC = ["start_ms", "duration_ms", "Grid X", "Grid Y", "Grid Z", "Block X", "Block Y", "Block Z",
                 "Registers Per Thread"," Static SMem", "Dynamic SMem", "Device", "Context", "Stream",
                 "transferred_data_byte", "Virtual Address", "name", "Correlation_ID"]
NVPROF_HEADER_NOMETRIC_FILTERED = NVPROF_HEADER_NOMETRIC[:2] + [NVPROF_HEADER_NOMETRIC[-2]]

# NVPROF_HEADER_METRIC = ["Device", "Context", "Stream", "name", "Correlation_ID",
#                         "dram_read_throughput", "dram_write_throughput", "dram_read_bytes", "dram_write_bytes", 
#                         "l2_global_atomic_store_bytes", "l2_global_load_bytes", "l2_global_reduction_bytes", "l2_local_global_store_bytes", "l2_local_load_bytes", "l2_read_throughput", "l2_write_throughput", 
#                         "inst_executed", "ipc", "flop_count_dp", "flop_count_sp"]
# NVPROF_HEADER_METRIC_FILTERED = [NVPROF_HEADER_METRIC[3]] + NVPROF_HEADER_METRIC[5:]

NVPROF_HEADER_METRIC = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Kernel Time", "Context", "Stream", "Section Name", "Metric Name", "Metric Unit", "Metric Value"]
NVPROF_HEADER_METRIC_FILTERED = [NVPROF_HEADER_METRIC[0], NVPROF_HEADER_METRIC[4], NVPROF_HEADER_METRIC[-3], NVPROF_HEADER_METRIC[-1]]

OPERATIONS_TO_MERGE = set(["htod", "dtoh"])

NUM_ITER = 30

# Maximum memory bandiwth, in GB/s. of the GPU (currently: GTX 1660);
MAX_GPU_BANDWIDTH = 336 
MAX_L2_GPU_BANDWIDTH = 450  # Not publicly known, estimated using nvvp;
GPU_CLOCK_HZ = 1_785_000_000
GPU_NUM_SM = 22

def load_data(b, p, files):
    
    # Associate each file to a key that represents its content;
    files_dict = {tuple(file.split(".")[0].split("_")[:4]): file for file in files}
    
    ##############################
    # Process file with execution time;
    ##############################
    
    input_file = os.path.join(DEFAULT_RES_DIR, INPUT_DATE, files_dict[(b, p, "nometric", "True")])
    data_nometric = pd.read_csv(input_file, skiprows=5, names=NVPROF_HEADER_NOMETRIC)
    header = pd.read_csv(input_file, skiprows=3, nrows=1)
    start_unit = header.iloc[0, 0]
    duration_unit = header.iloc[0, 1]
    
    # Keep only a subset of columns;
    data_nometric = data_nometric[NVPROF_HEADER_NOMETRIC_FILTERED]
    
    # Remove rows with NaN Duration;
    data_nometric = data_nometric.dropna(subset=["duration_ms"]).reset_index(drop=True)
    
    # Convert start and duration from seconds to milliseconds;
    if start_unit == "s":
        data_nometric["start_ms"] *= 1000
    elif start_unit == "us":
        data_nometric["start_ms"] /= 1000
    if duration_unit == "s":
        data_nometric["duration_ms"] *= 1000
    elif duration_unit == "us":
        data_nometric["duration_ms"] /= 1000
    
    # Set the start of the computation equal to 0;
    data_nometric["start_ms"] -= data_nometric["start_ms"].iloc[0]
       
    # Set the end of the computation;
    data_nometric["end_ms"] = data_nometric["duration_ms"] + data_nometric["start_ms"]
    
    # Clean names of operations;
    data_nometric["name"] = data_nometric["name"].replace({
        "[Unified Memory Memcpy HtoD]": "htod",
        "[Unified Memory Memcpy DtoH]": "dtoh",
        "[Unified Memory GPU page faults]": "pagefault",
        "[Unified Memory page throttle]": "throttle"
        })
    
    # Keep only kernel computations;
    data_nometric = data_nometric[~data_nometric["name"].isin(["htod", "dtoh", "pagefault", "throttle"])].reset_index(drop=True)
    
    # Keep just the name of kernels;
    data_nometric["name"] = data_nometric["name"].apply(lambda x: x.split("(")[0])
    
    ##############################
    # Process file with memory access information;
    ##############################
    
    input_file = os.path.join(DEFAULT_RES_DIR, INPUT_DATE, files_dict[(b, p, "metric", "True" if p == ASYNC_POLICY_NAME else "False")])
    print(b, p)
    data_metric = pd.read_csv(input_file, skiprows=3, names=NVPROF_HEADER_METRIC)
    # Keep only a subset of columns;
    data_metric = data_metric[NVPROF_HEADER_METRIC_FILTERED]
    data_metric = data_metric.fillna(0)

    # Keep only kernel computations;
    data_metric["Kernel Name"] = data_metric["Kernel Name"].apply(lambda x: x.split("(")[0])
    # Rename the "name" column to allow debugging after merging;
    data_metric = data_metric.rename(columns={"Kernel Name": "name_metric"})
    data_metric["Metric Value"] = data_metric["Metric Value"].str.replace(",", "").astype(float)

    # Pivot the table to obtain metrics for each kernel;
    data_metric = pd.pivot_table(data_metric, values="Metric Value", index=["ID", "name_metric"], columns="Metric Name").reset_index()
    
    # Create a new table with derived metrics;
    data_metric_2 = data_metric[["name_metric"]].copy()
    data_metric_2["dram_read_bytes"] = data_metric["dram__bytes_read.sum"]
    data_metric_2["dram_write_bytes"] = data_metric["dram__bytes_write.sum"]
    data_metric_2["l2_global_atomic_store_bytes"] = data_metric["lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_atom.sum"]
    data_metric_2["l2_global_load_bytes"] = data_metric["lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum"]
    data_metric_2["l2_global_reduction_bytes"] = 0
    data_metric_2["l2_local_global_store_bytes"] = data_metric["lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_local_op_st.sum"] + \
        data_metric["lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum"]
    data_metric_2["l2_local_load_bytes"] = data_metric["lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_local_op_ld.sum"]
    data_metric_2["flop_count_dp"] = data_metric["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"] + \
        data_metric["smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"] + \
        data_metric["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"] * 2
    data_metric_2["flop_count_sp"] = data_metric["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"] + \
        data_metric["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"] + \
        data_metric["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"] * 2
    data_metric_2["inst_executed"] = data_metric["smsp__inst_executed.sum"]
    data_metric_2["ipc"] = data_metric["smsp__inst_executed.avg.per_cycle_active"]
    
    # Turn bytes into GB;
    data_metric_2["dram_read_bytes"] /= 2**30
    data_metric_2["dram_write_bytes"] /= 2**30
    
    data_metric_2["l2_global_atomic_store_bytes"] /= 2**30
    data_metric_2["l2_global_load_bytes"] /= 2**30
    data_metric_2["l2_global_reduction_bytes"] /= 2**30
    
    data_metric_2["l2_local_global_store_bytes"] /= 2**30
    data_metric_2["l2_local_load_bytes"] /= 2**30
    
    data_metric_2["total_flop"] = data_metric_2["flop_count_dp"] + data_metric_2["flop_count_sp"]
    
    data_metric_2["total_l2_read_bytes"] = data_metric_2["l2_global_load_bytes"] + data_metric_2["l2_local_load_bytes"]
    data_metric_2["total_l2_write_bytes"] = data_metric_2["l2_global_atomic_store_bytes"] + data_metric_2["l2_global_reduction_bytes"] + data_metric_2["l2_local_global_store_bytes"]

    # Concatenate the 2 tables;
    data = pd.concat([data_nometric, data_metric_2], axis=1)

    # Look for inconsistencies;
    assert(len(data_metric_2) == len(data_nometric))
    # Note: this check can fail, as kernels with dependencies can be scheduled in different order from the sync kernels.
    # It doesn't matter for the memory throughput computation, as we consider the total execution time;
    # assert((data["name"] == data["name_metric"]).all())  

    # Check if throughput is close to the one computed by nvprof, for debugging.
    # This is relevant only for "sync" policies, as the execution times for the 2 tables are consistent;
    data["estimated_read_througput"] = data["dram_read_bytes"] / (data["duration_ms"] / 1000)
    data["estimated_write_througput"] = data["dram_write_bytes"] / (data["duration_ms"] / 1000)
    data["estimated_memory_througput"] = data["estimated_read_througput"] + data["estimated_write_througput"]
    data["estimated_l2_read_througput"] = data["total_l2_read_bytes"] / (data["duration_ms"] / 1000)
    data["estimated_l2_write_througput"] = data["total_l2_write_bytes"] / (data["duration_ms"] / 1000)
    data["estimated_l2_througput"] = data["estimated_l2_read_througput"] + data["estimated_l2_write_througput"]
    data["gigaflops"] = (data["total_flop"] / 10**9) / (data["duration_ms"] / 1000)
    
    data["estimated_ipc"] = data["inst_executed"] / (GPU_CLOCK_HZ * (data["duration_ms"] / 1000)) / GPU_NUM_SM
    
    # Add index columns;
    data["benchmark"] = b
    data["policy"] = p
    return data


def get_computation_time_with_overlap(data):
    """
    For each computation, look at the computations before it and compute the length of the overlap with them, in seconds.
    By definition, a computation has 0 overlap with itself;
    """
    curr_start = 0
    curr_end = 0
    total_duration = 0
    for i, r in data.iterrows():
        if r["start_ms"] < curr_end:
            curr_end = r["end_ms"]
        else:
            # Found the end of a contiguous computation segment;
            total_duration += curr_end - curr_start
            curr_start = r["start_ms"]
            curr_end = r["end_ms"]
    
    # Add the last computation;
    total_duration += curr_end - curr_start
        
    return total_duration


def autolabel(ax, rects1, rects2):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects2):
        height1 = rects1[i].get_height()
        height2 = rect.get_height()
        # ax.annotate('{:.2f}x'.format(height2 / height1),
        ax.annotate('{:.2f}x'.format(max(height2 / height1, 1)),
                    xy=(rect.get_x(), height2),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7)
        
def barplot(data, ax, title, y_column, y_limit, annotation_title, y_ticks=6, y_tick_format=lambda l: f"{l:.2f}", baseline_annotation_format=lambda l: f"{l:.2f}"):
    
    # Obtain x values for the plot;
    x = np.arange(len(data["benchmark"].unique()))

    # Obtain labels;
    x_labels = [BENCHMARK_NAMES[l] for l in data["benchmark"].unique()]

    peach = "#fab086"
    green = "#6cb77c"
    palette = [peach, green]
    edgecolor = "#2f2f2f"
    
    bar_width = 0.35
    
    # Obtain y;
    y_sync = data[data["policy"] == "sync"][y_column]
    y_async = data[data["policy"] == ASYNC_POLICY_NAME][y_column]

    rects1 = ax.bar(x - bar_width / 2, y_sync, bar_width, label="sync", color=palette[0], edgecolor=edgecolor)
    rects2 = ax.bar(x + bar_width / 2, y_async, bar_width, label=ASYNC_POLICY_NAME, color=palette[1], edgecolor=edgecolor)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8, va="center")
    
    # ax.set_ylim((0, 1.1 * summary["memory_throughput"].max()))
    ax.set_ylim(y_limit)
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(y_ticks))
    ax.set_yticklabels(labels=[y_tick_format(l) for l in ax.get_yticks()], ha="right", fontsize=8)
    ax.grid(True, axis="y")
    
    # ax.annotate(title, fontsize=9, x=.02, y=0.95, ha="left")
    plt.suptitle("Hardware metrics for each\nbenchmark and execution policy,\nGTX 1660 Super", fontsize=14, x=.01, y=0.99, ha="left")
    ax.annotate(title, xy=(0, 1.08), fontsize=10, ha="left", xycoords="axes fraction")#, xycoords="data", xytext=(0, 100), textcoords="offset points")
    autolabel(ax, rects1, rects2)
    
    # Add baseline annotations;
    for i, b in enumerate(BENCHMARK_NAMES):
        position = x[i]
        serial_throughput = summary[(summary["benchmark"] == b) & (summary["policy"] == "sync")][y_column].iloc[0]
        if i == 0: 
            ax.annotate(annotation_title, xy=(0, 0), fontsize=9, ha="left", va="center", xycoords="data", xytext=(-32, -20), textcoords="offset points")
        ax.annotate(baseline_annotation_format(serial_throughput), xy=(position - bar_width, 0), fontsize=9, ha="center", va="center", xycoords="data", color=palette[0], xytext=(7, -30), textcoords="offset points")
    
    # Legend;  
    labels = [POLICIES_DICT[p] for p in POLICIES]
    custom_lines = [Patch(facecolor=palette[i], edgecolor="#2f2f2f", label=l)
                    for i, l in enumerate(labels)]
    leg = fig.legend(custom_lines, labels, bbox_to_anchor=(1, 1), fontsize=10, ncol=1)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')        


if __name__ == "__main__":
    
    files = os.listdir(os.path.join(DEFAULT_RES_DIR, INPUT_DATE))
       
    output_res = []
    for b in BENCHMARK_NAMES.keys():
        for p in POLICIES:
            output_res += [load_data(b, p, files)]
            
    # Create a single table;
    res = pd.concat(output_res, ignore_index=True)
    # Sort columns;
    res = res[list(res.columns[-2:]) + [res.columns[2]] + [res.columns[0]] + [res.columns[3]] + [res.columns[1]] + list(res.columns[5:-2])]
    
    # For each benchmark and policy, compute the total computation time;
    total = []
    summary_list = []
    for (b, p), group in res.groupby(by=["benchmark", "policy"], sort=False):
        total += [group]
        overlap_computation_time = get_computation_time_with_overlap(group)
        print(b, p, f"{overlap_computation_time:.2f}")
        
        # Device memory;
        total_memory_accessed = group["dram_read_bytes"].sum() + group["dram_write_bytes"].sum()
        memory_throughput = total_memory_accessed / (overlap_computation_time / 1000)
        
        # L2 cache;
        total_l2_accessed = group["total_l2_read_bytes"].sum() + group["total_l2_write_bytes"].sum()
        l2_throughput = total_l2_accessed / (overlap_computation_time / 1000)
        
        # IPC;
        total_instructions = group["inst_executed"].sum() 
        ipc = total_instructions / (GPU_CLOCK_HZ * (overlap_computation_time / 1000)) / GPU_NUM_SM
        
        # GigaFLOPS;
        total_flop = group["total_flop"].sum() 
        gigaflops = (total_flop / 10**9) / (overlap_computation_time / 1000)
        
        print(total_memory_accessed, total_l2_accessed)
        
        summary_list += [[b, p, overlap_computation_time, total_memory_accessed, memory_throughput, memory_throughput / MAX_GPU_BANDWIDTH, l2_throughput, l2_throughput / MAX_L2_GPU_BANDWIDTH,  ipc, gigaflops]]
    data = pd.concat(total)   
    summary = pd.DataFrame(summary_list, columns=["benchmark", "policy", "duration_ms", "dram_accessed_GB", "memory_throughput", "max_memory_throughput_perc", "l2_throughput", "max_l2_throughput_perc", "ipc", "gigaflops"])
    
    #%% Create barplot with memory throughput;   
    
    sns.set_style("white", {"ytick.left": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 25 
    plt.rcParams['axes.labelpad'] = 9 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    plt.rcParams['xtick.major.pad'] = 5
    
    num_col = 2
    num_rows = 2
    
    fig, axes = plt.subplots(num_rows, num_col, figsize=(2.4 * num_col, 2.4 * num_rows)) 
    plt.subplots_adjust(top=0.80,
                    bottom=0.10,
                    left=0.13,
                    right=.99,
                    hspace=0.6,
                    wspace=0.4)
   
    barplot(summary, axes[0, 0], "Device memory throughput",
            "memory_throughput", (0, 120), "Serial throughput (GB/s):", y_ticks=7, y_tick_format=lambda l: f"{int(l)} GB/s", baseline_annotation_format=lambda l: f"{int(l)}")
    barplot(summary, axes[0, 1], "L2 cache throughput",
            "l2_throughput", (0, 150), "Serial throughput (GB/s):", y_ticks=6, y_tick_format=lambda l: f"{int(l)} GB/s", baseline_annotation_format=lambda l: f"{int(l)}")
    barplot(summary, axes[1, 0], "IPC",
            "ipc", (0, 1.0), "Serial IPC:", y_ticks=6, y_tick_format=lambda l: f"{l:.2f}", baseline_annotation_format=lambda l: f"{l:.2f}")
    barplot(summary, axes[1, 1], "GFLOPS32/64",
            "gigaflops", (0, 120), "GFLOPS32/64:", y_ticks=7, y_tick_format=lambda l: f"{int(l)}", baseline_annotation_format=lambda l: f"{int(l)}")
    
    save_plot(PLOT_DIR, "memory_throughput_{}.{}", OUTPUT_DATE)
    
    #%%
    tmp = res[res["policy"] == "sync"].groupby(by=["benchmark", "policy", "name"]).mean()
    tmp["ipc_fix"] = tmp["estimated_ipc"] / 22
    tmp["ipc_perc"] = ( tmp["ipc_fix"] -  tmp["ipc"]) /  tmp["ipc"]
    
    print(np.median(tmp["ipc_perc"]))
