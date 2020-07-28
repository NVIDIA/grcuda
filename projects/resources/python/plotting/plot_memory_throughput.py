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


DEFAULT_RES_DIR = "../../../../data/nvprof_log"

INPUT_DATE = "2020_07_22"
OUTPUT_DATE = "2020_07_28"
PLOT_DIR = "../../../../data/plots"

BENCHMARK_NAMES = {
    "b1": "Vector\nSquares",
    "b5": "B&S", 
    "b6": "ML\nEnsemble",
    "b7": "HITS", 
    "b8": "Images"
    }
POLICIES = ["sync", "default"]
POLICIES_DICT = {"default": "DAG Scheduling", "sync": "Serial Scheduling"}

NVPROF_HEADER_NOMETRIC = ["start_ms", "duration_ms", "Grid X", "Grid Y", "Grid Z", "Block X", "Block Y", "Block Z",
                 "Registers Per Thread"," Static SMem", "Dynamic SMem", "Device", "Context", "Stream",
                 "transferred_data_byte", "Virtual Address", "name", "Correlation_ID"]
NVPROF_HEADER_NOMETRIC_FILTERED = NVPROF_HEADER_NOMETRIC[:2] + [NVPROF_HEADER_NOMETRIC[-2]]

NVPROF_HEADER_METRIC = ["Device", "Context", "Stream", "name", "Correlation_ID", "dram_read_throughput",
                        "dram_read_transactions", "dram_read_bytes", "dram_write_bytes"]
NVPROF_HEADER_METRIC_FILTERED = [NVPROF_HEADER_METRIC[3]] + [NVPROF_HEADER_METRIC[5]] + NVPROF_HEADER_METRIC[7:]

OPERATIONS_TO_MERGE = set(["htod", "dtoh"])

NUM_ITER = 30

# Maximum memory bandiwth, in GB/s. of the GPU (currently: GTX 960);
MAX_GPU_BANDWIDTH = 112


def load_data(b, p, files):
    ##############################
    # Process file with execution time;
    ##############################
    
    input_file = os.path.join(DEFAULT_RES_DIR, INPUT_DATE, files_dict[(b, p, "nometric")])
    data_nometric = pd.read_csv(input_file, skiprows=5, names=NVPROF_HEADER_NOMETRIC)
    
    # Keep only a subset of columns;
    data_nometric = data_nometric[NVPROF_HEADER_NOMETRIC_FILTERED]
    
    # Remove rows with NaN Duration;
    data_nometric = data_nometric.dropna(subset=["duration_ms"]).reset_index(drop=True)
    
    # Convert start from seconds to milliseconds;
    data_nometric["start_ms"] *= 1000
    
    # Set the start of the computation equal to 0;
    data_nometric["start_ms"] -= data_nometric["start_ms"].iloc[0]
       
    # Set the end of the computation;
    data_nometric["end_ms"] = data_nometric["duration_ms"] + data_nometric["start_ms"]
    
    # Clean names of operations;
    data_nometric["name"] = data_nometric["name"].replace({
        "[Unified Memory Memcpy HtoD]": "htod",
        "[Unified Memory Memcpy DtoH]": "dtoh"
        })
    
    # Keep only kernel computations;
    data_nometric = data_nometric[~data_nometric["name"].isin(["htod", "dtoh"])].reset_index(drop=True)
    
    # Keep just the name of kernels;
    data_nometric["name"] = data_nometric["name"].apply(lambda x: x.split("(")[0])
    
    ##############################
    # Process file with memory access information;
    ##############################
    
    input_file = os.path.join(DEFAULT_RES_DIR, INPUT_DATE, files_dict[(b, p, "metric")])
    data_metric = pd.read_csv(input_file, skiprows=6, names=NVPROF_HEADER_METRIC)
    # Keep only a subset of columns;
    data_metric = data_metric[NVPROF_HEADER_METRIC_FILTERED]
    
    # Keep only kernel computations;
    data_metric["name"] = data_metric["name"].apply(lambda x: x.split("(")[0])
    
    # Rename the "name" column to allow debugging after merging;
    data_metric = data_metric.rename(columns={"name": "name_metric"})
    
    # Turn bytes into GB;
    data_metric["dram_read_bytes"] /= 2**30
    data_metric["dram_write_bytes"] /= 2**30
    
    # Concatenate the 2 tables;
    data = pd.concat([data_nometric, data_metric], axis=1)
    
    # Look for inconsistencies;
    assert(len(data_metric) == len(data_nometric))
    # Note: this check can fail, as kernels with dependencies can be scheduled in different order from the sync kernels.
    # It doesn't matter for the memory throughput computation, as we consider the total execution time;
    # assert((data["name"] == data["name_metric"]).all())  

    # Check if read throughput is close to the one computed by nvprof, for debugging.
    # This is relevant only for "sync" policies, as the execution times for the 2 tables are consistent;
    data["estimated_read_througput"] = data["dram_read_bytes"] / (data["duration_ms"] / 1000)
    
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


if __name__ == "__main__":
    
    files = os.listdir(os.path.join(DEFAULT_RES_DIR, INPUT_DATE))
    
    # Associate each file to a key that represents its content;
    files_dict = {tuple(file.split("_")[:3]): file for file in files}
    
    output_res = []
    for b in BENCHMARK_NAMES.keys():
        for p in POLICIES:
            output_res += [load_data(b, p, files)]
            
    # Create a single table;
    res = pd.concat(output_res, ignore_index=True)
    # Sort columns;
    res = res[list(res.columns[-2:]) + [res.columns[2]] + [res.columns[0]] + [res.columns[3]] + [res.columns[1]] + list(res.columns[5:9])]
    
    # For each benchmark and policy, compute the total computation time;
    summary_list = []
    for (b, p), group in res.groupby(by=["benchmark", "policy"]):
        overlap_computation_time = get_computation_time_with_overlap(group)
        total_memory_accessed = group["dram_read_bytes"].sum() + group["dram_write_bytes"].sum()
        memory_throughput = total_memory_accessed / (overlap_computation_time / 1000)
        summary_list += [[b, p, overlap_computation_time, total_memory_accessed, memory_throughput, memory_throughput / MAX_GPU_BANDWIDTH]]
        
    summary = pd.DataFrame(summary_list, columns=["benchmark", "policy", "duration_ms", "dram_accessed_GB", "memory_throughput", "max_memory_throughput_perc"])
    
    #%% Create barplot with memory throughput;
    
    # Obtain x values for the plot;
    x = np.arange(len(summary["benchmark"].unique()))

    # Obtain labels;
    x_labels = [BENCHMARK_NAMES[l] for l in summary["benchmark"].unique()]
    # Obtain y;
    y_sync = summary[summary["policy"] == "sync"]["memory_throughput"]
    y_default = summary[summary["policy"] == "default"]["memory_throughput"]
    
    sns.set_style("white", {"ytick.left": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 25 
    plt.rcParams['axes.labelpad'] = 9 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    plt.rcParams['xtick.major.pad'] = 10
    
    num_col = 1
    
    fig = plt.figure(figsize=(3 * num_col, 2.5)) 
    ax = fig.add_subplot()
    plt.subplots_adjust(top=0.75,
                    bottom=0.25,
                    left=0.15,
                    right=.99,
                    hspace=0.9,
                    wspace=0.0)
    palette = [COLORS["peach1"], COLORS["b8"]]
    edgecolor = "#2f2f2f"
    # p = ["#FFEDAB", "#FFDB8C", "#FFC773", "#FFAF66"]
    # p = ["#C8FCB6", "#96DE9B", "#66B784", "#469E7B"]
    
    bar_width = 0.35
    
    rects1 = ax.bar(x - bar_width / 2, y_sync, bar_width, label="sync", color=palette[0], edgecolor=edgecolor)
    rects2 = ax.bar(x + bar_width / 2, y_default, bar_width, label="default", color=palette[1], edgecolor=edgecolor)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8, va="center")
    
    # ax.set_ylim((0, 1.1 * summary["memory_throughput"].max()))
    ax.set_ylim((0, 50))
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(6))
    ax.set_yticklabels(labels=[f"{int(l)} GB/s" for l in ax.get_yticks()], ha="right", fontsize=6)
    ax.grid(True, axis="y")
    
    plt.suptitle("Device memory throughput\nfor each benchmark\nand execution policy", fontsize=9, x=.02, y=0.95, ha="left")
    
    def autolabel(rects1, rects2):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for i, rect in enumerate(rects2):
            height1 = rects1[i].get_height()
            height2 = rect.get_height()
            ax.annotate('{:.2f}x'.format(height2 / height1),
                        xy=(rect.get_x() + rect.get_width() / 2, height2),
                        xytext=(0, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)

    autolabel(rects1, rects2)
    
    # Add baseline annotations;
    for i, b in enumerate(BENCHMARK_NAMES):
        position = x[i]
        serial_throughput = summary[(summary["benchmark"] == b) & (summary["policy"] == "sync")]["memory_throughput"].iloc[0]
        ax.annotate(f"Serial\nthroughput: ", xy=(position, 0), fontsize=6, ha="center", xycoords="data", xytext=(0, -36), textcoords="offset points")
        ax.annotate(f"{int(serial_throughput)} GB/s", xy=(position, 0), fontsize=6, ha="center", xycoords="data", color="#469E7B", xytext=(0, -43.5), textcoords="offset points")
    
    # Legend;  
    labels = [POLICIES_DICT[p] for p in POLICIES]
    custom_lines = [Patch(facecolor=palette[i], edgecolor="#2f2f2f", label=l)
                    for i, l in enumerate(labels)]
    leg = fig.legend(custom_lines, labels, bbox_to_anchor=(1, 0.96), fontsize=8, ncol=1)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    save_plot(PLOT_DIR, "memory_throughput_{}.{}", OUTPUT_DATE)
    