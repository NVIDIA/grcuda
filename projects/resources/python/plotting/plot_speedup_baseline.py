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
from plot_utils import COLORS, get_exp_label, get_ci_size, save_plot


INPUT_DATE = "2020_07_10_15_42_44_grcuda"
OUTPUT_DATE = "2020_07_10"
PLOT_DIR = "../../../../data/plots"

BENCHMARK_NAMES = {"b1": "Vector Squares", "b5": "B&S", "b6": "ML Ensemble", "b7": "HITS", "b8": "Images"}

def build_exec_time_plot(data, gridspec, x, y):
    
    data["size_str"] = data["size"].astype(str)
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)
    
    ax = sns.lineplot(x="size_str", y="computation_speedup", data=data, color=COLORS["bb1"], ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=False, sort=False, ci=None, zorder=2)
    
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

    # Set the same y limits in each plot;
    ax.set_ylim((0, 3))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=45, ha="right", fontsize=9, rotation_mode="anchor")
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(7))
    if y == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=12)
    else:
        ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
        # Hide tick markers;
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False) 
            tic.tick2line.set_visible(False) 
    
    ax.set_ylabel(None)     
    ax.set_xlabel(None) 
    
    # Add benchmark name and baseline execution time annotations;
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.1), fontsize=14, ha="center", xycoords="axes fraction")
    ax.annotate(f"Baseline exec. time (ms):", xy=(0, -0.37), fontsize=9, ha="left", xycoords="axes fraction", color=COLORS["r4"])
    
    for i, l in enumerate(labels):
        baseline_median = np.median(data[data["size"] == int(l)]["baseline_time_sec"])
        ax.annotate(f"{int(1000 * baseline_median)}", xy=(i,  -0.47), fontsize=9, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    # Add block size annotation;
    if y == 0:
        ax.annotate(f"Block size:\n1D={data['block_size_1d'].iloc[0]}, 2D={data['block_size_2d'].iloc[0]}x{data['block_size_2d'].iloc[0]}", xy=(-0.65, 1.25), fontsize=14, ha="left", xycoords="axes fraction") 
    
    # Turn off tick lines;
    ax.xaxis.grid(False)
    
    return ax


def build_exec_time_plot_1_row(data, gridspec, y):
    
    data["size_str"] = data["size"].astype(str)
    
    palette = [COLORS["peach1"], COLORS["b8"], COLORS["b2"], COLORS["b4"]]
    markers = ["o", "X", "D", "P"]
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[0, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)
    ax = sns.lineplot(x="size_str", y="computation_speedup", hue="block_size_str", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, zorder=2)
    data_averaged = data.groupby(["size_str", "block_size_str"], as_index=True)["computation_speedup"].apply(gmean).reset_index()
    order = data["block_size_str"].unique()
    ax = sns.scatterplot(x="size_str", y="computation_speedup", hue="block_size_str", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="block_size_str", hue_order=order, style_order=order, linewidth=0.05)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Set the same y limits in each plot;
    ax.set_ylim((0.8, 1.5))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=0, ha="center", fontsize=9)
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(8))
    if y == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=12)
    else:
        ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
        # Hide tick markers;
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False) 
            tic.tick2line.set_visible(False) 
    
    ax.set_ylabel(None)     
    ax.set_xlabel(None) 
    
    # Add benchmark name and baseline execution time annotations;
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.1), fontsize=14, ha="center", xycoords="axes fraction")
    
     # Turn off tick lines;
    ax.xaxis.grid(False)
    
    # Add baseline execution time annotations (median of execution time across blocks);
    ax.annotate(f"Median baseline exec. time (ms):", xy=(0, -0.2), fontsize=9, ha="left", xycoords="axes fraction", color=COLORS["r4"])
    for i, l in enumerate(labels):
        baseline_median = np.median(data[data["size"] == int(l)]["baseline_time_sec"])
        ax.annotate(f"{int(1000 * baseline_median)}", xy=(i,  -0.27), fontsize=9, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    # Legend;   
    if y == 0:
        legend_labels = [f"1D={x.split(',')[0]}, 2D={x.split(',')[1]}" for x in data["block_size_str"].unique()]
        custom_lines = [
            lines.Line2D([], [], color="white", marker=markers[i], markersize=10, label=legend_labels[i], markerfacecolor=palette[i], markeredgecolor="#2f2f2f") 
            for i in range(len(legend_labels))]
        
        leg = fig.legend(custom_lines, legend_labels,
                                 bbox_to_anchor=(0.955, 0.94), fontsize=12, ncol=len(legend_labels), handletextpad=0.1)
        leg.set_title("Block size:")
        leg._legend_box.align = "left"
    
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
    fig = plt.figure(figsize=(2.5 * num_col, 4 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.8,
                    bottom=0.15,
                    left=0.2,
                    right=0.90,
                    hspace=1.1,
                    wspace=0.15)
        
    exec_time_axes = []
    for b_i, b in enumerate(benchmark_list):
        for block_size_i, block_size in enumerate(block_size_list): 
            curr_res = data[(data["benchmark"] == b) & (data["block_size_str"] == block_size)].reset_index(drop=True)  
            exec_time_axes += [build_exec_time_plot(curr_res, gs, block_size_i, b_i)]
            
    plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=20, ha="center", va="center", xycoords="figure fraction")
    plt.annotate("Speedup over serial scheduling", xy=(0.02, 0.5), fontsize=20, ha="center", va="center", rotation=90, xycoords="figure fraction")    
    plt.suptitle("Execution time speedup\nover serial kernel scheduling", fontsize=25, x=.05, y=0.99, ha="left")
    
    save_plot(PLOT_DIR, "speedup_baseline_{}.{}", OUTPUT_DATE)
    
    #%% Similar plot, but all block sizes are on 1 row;
    
    
    sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    plt.rcParams["font.family"] = ["Latin Modern Roman"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    # Lists of benchmarks and block sizes;
    benchmark_list = sorted(data["benchmark"].unique())
    num_col = len(benchmark_list)
    num_row = 1
    fig = plt.figure(figsize=(2.6 * num_col, 4.1 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.65,
                    bottom=0.21,
                    left=0.1,
                    right=0.95,
                    hspace=1.1,
                    wspace=0.15)
        
    exec_time_axes = []
    for b_i, b in enumerate(benchmark_list):
        curr_res = data[data["benchmark"] == b].reset_index(drop=True)  
        exec_time_axes += [build_exec_time_plot_1_row(curr_res, gs, b_i)]
        
    plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    plt.annotate("Speedup over\nserial scheduling", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    plt.suptitle("Execution time speedup\nover serial kernel scheduling", fontsize=20, x=.05, y=0.92, ha="left")

    save_plot(PLOT_DIR, "speedup_baseline_1_row_{}.{}", OUTPUT_DATE)
    
    