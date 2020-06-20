#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:14:30 2020

@author: alberto.parravicini
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy.stats.mstats import gmean
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.lines as lines

import os
from load_data import load_data
from plot_utils import COLORS, get_exp_label, get_ci_size


INPUT_DATE = "2020_06_20_16_34_27"
OUTPUT_DATE = "2020_06_20"
PLOT_DIR = "../../../../data/plots"

BENCHMARK_NAMES = {"b1": "Vector Squares", "b6": "ML Ensemble", "b7": "HITS", "b8": "Images"}

def build_exec_time_plot(data, gridspec, x, y):
    
    data["size_str"] = data["size"].astype(str)
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax = sns.lineplot(x="size_str", y="computation_speedup", data=data, color=COLORS["bb1"], ax=ax, estimator=gmean,
                      err_style="bars", linewidth=3, legend=False, sort=False, ci=None, zorder=2)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Add rectangles to represent variance
    rectangles = []
    for s_i, s in enumerate(labels):
        curr_data = data[data["size"] == s]
        upper_ci_size, lower_ci_size, center = get_ci_size(curr_data["computation_speedup"], estimator=gmean, ci=0.90)
        bottom = center - lower_ci_size
        width = 0.1
        lower_left = [s_i - width / 2, bottom]
        # Add an offset to the x position, to avoid overlapping;
        rectangles += [Rectangle(lower_left, width, upper_ci_size + lower_ci_size)]
        
    pc = PatchCollection(rectangles, facecolor="white", edgecolor="#2f2f2f", linewidth=0.5, zorder=3, clip_on=True, alpha=0.7)         
    ax.add_collection(pc)         


    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=4, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=0, ha="center", fontsize=15)
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(5))
    ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=15)
    
    ax.set_ylabel(None)     
    ax.set_xlabel(None) 
    
    # Add benchmark name and baseline execution time annotations;
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.1), fontsize=14, ha="center", xycoords="axes fraction")
    ax.annotate(f"Baseline exec. time:", xy=(0, -0.35), fontsize=12, ha="left", xycoords="axes fraction", color=COLORS["r4"])
    
    for i, l in enumerate(labels):
        baseline_median = np.median(data[data["size"] == int(l)]["baseline_time_sec"])
        ax.annotate(f"{1000 * baseline_median:.1f} ms", xy=(i,  -0.47), fontsize=12, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    # Add block size annotation;
    if y == 0:
        ax.annotate(f"Block size:\n1D={data['block_size_1d'].iloc[0]}, 2D={data['block_size_2d'].iloc[0]}x{data['block_size_2d'].iloc[0]}", xy=(-0.65, 1.25), fontsize=14, ha="left", xycoords="axes fraction") 
    
    # Turn off tick lines;
    ax.xaxis.grid(False)
    
    return ax
    

if __name__ == "__main__":
    data = load_data(INPUT_DATE, skip_iter=3)
    
    # Ignore synchronous execution;
    data = data[data["exec_policy"] != "sync"]
    
    sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    plt.rcParams["font.family"] = ["Latin Modern Roman"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    # Lists of benchmarks and block sizes;
    benchmark_list = sorted(data["benchmark"].unique())
    block_size_list = sorted(data["block_size_str"].unique(), key=lambda x: [int(y) for y in x.split(",")])
    num_col = len(benchmark_list)
    num_row = len(block_size_list)
    fig = plt.figure(figsize=(4 * num_col, 4 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.8,
                    bottom=0.2,
                    left=0.13,
                    right=0.90,
                    hspace=1.1,
                    wspace=0.8)
        
    exec_time_axes = []
    for b_i, b in enumerate(benchmark_list):
        for block_size_i, block_size in enumerate(block_size_list): 
            curr_res = data[(data["benchmark"] == b) & (data["block_size_str"] == block_size)].reset_index(drop=True)  
            exec_time_axes += [build_exec_time_plot(curr_res, gs, block_size_i, b_i)]
            
    plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=20, ha="center", va="center", xycoords="figure fraction")
    plt.annotate("Speedup over serial execution", xy=(0.02, 0.5), fontsize=20, ha="center", va="center", rotation=90, xycoords="figure fraction")    
    plt.suptitle("Execution Time Speedup\nw.r.t Serial Baseline", fontsize=25, x=.05, y=0.99, ha="left")
    
    plt.savefig(os.path.join(PLOT_DIR, f"speedup_baseline_{OUTPUT_DATE}.pdf"), dpi=300)
    
    