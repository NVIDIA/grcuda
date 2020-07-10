#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 15:26:36 2020

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
from load_data import load_data, load_data_cuda, join_tables
from plot_utils import COLORS, get_exp_label, get_ci_size, save_plot
import matplotlib.ticker as ticker


##############################
##############################


INPUT_DATE_GRCUDA = "2020_07_10_15_42_44_grcuda"
INPUT_DATE_CUDA = "2020_07_10_16_18_02_cuda"
OUTPUT_DATE = "2020_07_10"
PLOT_DIR = "../../../../data/plots"

BENCHMARK_NAMES = {"b1": "Vector Squares", "b5": "B&S", "b6": "ML Ensemble", "b7": "HITS", "b8": "Images"}

##############################
##############################

def build_exec_time_plot_grcuda_cuda(data, gridspec, x, y):
    
    palette = [COLORS["peach1"], COLORS["bb1"]]
    markers = ["o"] * len(palette)
    
    data["size_str"] = data["size"].astype(str)
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)

    ax = sns.lineplot(x="size_str", y="grcuda_cuda_speedup", hue="exec_policy", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, ci=None, sort=False, zorder=2)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Add rectangles to represent variance;
    for p_i, p in enumerate(sorted(data["exec_policy"].unique())):
        rectangles = []
        for s_i, s in enumerate(labels):
            curr_data = data[(data["size"] == s) & (data["exec_policy"] == p)]
            upper_ci_size, lower_ci_size, center = get_ci_size(curr_data["grcuda_cuda_speedup"], estimator=gmean, ci=0.90)
            bottom = center - lower_ci_size
            width = 0.1
            lower_left = [s_i - width / 2, bottom]
            # Add an offset to the x position, to avoid overlapping;
            lower_left[0] += (2 * p_i - 1) * (width / 3.5)
            rectangles += [Rectangle(lower_left, width, upper_ci_size + lower_ci_size)]
            
        pc = PatchCollection(rectangles, facecolor=palette[p_i], edgecolor="#2f2f2f", linewidth=0.5, zorder=3, clip_on=True, alpha=0.7)         
        ax.add_collection(pc)         

    # Set the same y limits in each plot;
    ax.set_ylim((0, 2))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=45, ha="right", fontsize=9, rotation_mode="anchor")
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(5))
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
    ax.annotate(f"GrCUDA serial time (ms):", xy=(0, -0.37), fontsize=9, ha="left", xycoords="axes fraction", color=COLORS["r4"])
    
    for i, l in enumerate(labels):
        baseline_median = np.median(data[data["size"] == int(l)]["baseline_time_sec_grcuda"])
        ax.annotate(f"{int(1000 * baseline_median)}", xy=(i,  -0.47), fontsize=9, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    # Add block size annotation;
    if y == 0:
        ax.annotate(f"Block size:\n1D={data['block_size_1d'].iloc[0]}, 2D={data['block_size_2d'].iloc[0]}x{data['block_size_2d'].iloc[0]}", xy=(-0.65, 1.25), fontsize=14, ha="left", xycoords="axes fraction") 
    
    # Turn off tick lines;
    ax.xaxis.grid(False)
    
    # Legend;
    if y == 0 and x == 0:
        legend_labels = ["DAG Scheduling", "Serial Scheduling"]
        custom_lines = [
            lines.Line2D([], [], color="white", marker=markers[i], markersize=10, label=legend_labels[i], markerfacecolor=palette[i], markeredgecolor="#2f2f2f") 
            for i in range(len(legend_labels))]
        
        leg = fig.legend(custom_lines, legend_labels,
                                 bbox_to_anchor=(0.91, 0.98), fontsize=12, ncol=1, handletextpad=0.1)
        leg.set_title(None)
        leg._legend_box.align = "left"
        
    
    return ax


def build_exec_time_plot_grcuda_cuda_compact(data, gridspec, x, y):
    
    data["size_str"] = data["size"].astype(str)
    
    legend_labels = ["DAG Scheduling", "Serial Scheduling"]
    
    palette = [COLORS["peach1"], COLORS["b8"], COLORS["b2"], COLORS["b4"]][:len(data["block_size_str"].unique())]
    markers = ["o", "X", "D", "P"][:len(data["block_size_str"].unique())]
    order = data["block_size_str"].unique()
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)

    ax = sns.lineplot(x="size_str", y="grcuda_cuda_speedup", hue="block_size_str", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, hue_order=order, zorder=2)
    data_averaged = data.groupby(["size_str", "block_size_str"], as_index=True)["grcuda_cuda_speedup"].apply(gmean).reset_index()
    
    ax = sns.scatterplot(x="size_str", y="grcuda_cuda_speedup", hue="block_size_str", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="block_size_str", hue_order=order, style_order=order, linewidth=0.05)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Set the same y limits in each plot;
    ax.set_ylim((0, 2))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=0, ha="center", fontsize=9)
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(5))
    if y == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=12)
    else:
        ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
        # Hide tick markers;
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False) 
            tic.tick2line.set_visible(False) 
            
    # Add policy annotation;
    if y == 0:
        ax.annotate(f"{legend_labels[x]}", xy=(-0.15, 1.25), fontsize=14, ha="left", xycoords="axes fraction") 
    
    ax.set_ylabel(None)     
    ax.set_xlabel(None) 
    
    # Add benchmark name and baseline execution time annotations;
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.1), fontsize=14, ha="center", xycoords="axes fraction")
    
     # Turn off tick lines;
    ax.xaxis.grid(False)
    
    # Add baseline execution time annotations (median of execution time across blocks);
    ax.annotate(f"Median baseline exec. time (ms):", xy=(0, -0.22), fontsize=9, ha="left", xycoords="axes fraction", color=COLORS["r4"])
    for i, l in enumerate(labels):
        baseline_median = np.median(data[data["size"] == int(l)]["baseline_time_sec_cuda"])
        ax.annotate(f"{int(1000 * baseline_median)}", xy=(i,  -0.29), fontsize=9, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    # Legend;   
    if x == 0 and y == 0:
        legend_labels = [f"1D={x.split(',')[0]}, 2D={x.split(',')[1]}" for x in data["block_size_str"].unique()]
        custom_lines = [
            lines.Line2D([], [], color="white", marker=markers[i], markersize=10, label=legend_labels[i], markerfacecolor=palette[i], markeredgecolor="#2f2f2f") 
            for i in range(len(legend_labels))]
        
        leg = fig.legend(custom_lines, legend_labels,
                                 bbox_to_anchor=(0.95, 1), fontsize=12, ncol=len(legend_labels) // 2, handletextpad=0.1)
        leg.set_title("Block size:")
        leg._legend_box.align = "left"
    
    return ax


def ridgeplot(data):
    # Plotting setup;
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = ["Latin Modern Roman"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    
    # Plot only some data, for now;
    data = data[(data["benchmark"] == "b8") & (data["exec_policy"] == "default")].copy()
    
    # Normalize execution times so that the CUDA baseline has median 1;
    data["normalized_time_cuda"] = 1
    data["normalized_time_grcuda"] = 1
    
    block_sizes = data["block_size_str"].unique()
    sizes = data["size"].unique()
    
    grouped_data = data.groupby(["block_size_str", "size"], as_index=False)
    for group_key, group in grouped_data:
        # Compute the median baseline computation time;
        median_baseline = np.median(group["computation_sec_cuda"])
        # Compute the speedup for this group;
        data.loc[group.index, "normalized_time_cuda"] = group["computation_sec_cuda"].values / median_baseline
        data.loc[group.index, "normalized_time_grcuda"] = group["computation_sec_grcuda"].values / median_baseline
            
    # Initialize the plot;
    g = sns.FacetGrid(data, row="size", aspect=5, height=1, sharey=False, col="block_size_str")

    # Plot a vertical line corresponding to speedup = 1;
    g.map(plt.axvline, x=1, lw=0.75, clip_on=True, zorder=0, linestyle="--", ymax=0.5)         
    # Plot the densities. Plot them twice as the second time we plot just the white contour;     
                                                                            
    g.map(sns.kdeplot, "normalized_time_cuda", clip_on=False, shade=True, alpha=0.6, lw=1, color=COLORS["peach1"], zorder=2)  
    g.map(sns.kdeplot, "normalized_time_grcuda", clip_on=False, shade=True, alpha=0.6, lw=1, color=COLORS["b8"], zorder=2)
    g.map(sns.kdeplot, "normalized_time_cuda", clip_on=False, color="w", lw=1.1, zorder=2)
    g.map(sns.kdeplot, "normalized_time_grcuda", clip_on=False, color="w", lw=1.1, zorder=2)
    # Plot the horizontal line below the densities;
    g.map(plt.axhline, y=0, lw=0.75, clip_on=False, zorder=5, color="0.6")
    
    # Fix the horizontal axes so that they are between 0.5 and 1.25;
    def set_x_width(label="", color="#2f2f2f"):
        ax = plt.gca()
        ax.set_xlim(left=0.8, right=1.2)
    g.map(set_x_width)
    
    # Titles and labels;
    g.set_titles("")
    g.set(xlabel=None)
    
    # Add block size labels;
    for i, ax in enumerate(g.axes[-1]):
        ax.annotate("1D={}, 2D={}".format(*block_sizes[i].split(",")), xy=(0.5, -0.8), xycoords="axes fraction", ha="center", color="#2f2f2f", fontsize=14)      
    for i, ax in enumerate(g.axes[:, 0]):
        ax.annotate(f"{get_exp_label(sizes[i])}", xy=(-0.1, 0.05), xycoords="axes fraction", ha="center", color="#2f2f2f", fontsize=14)      
    
    # Fix the borders. This must be done here as the previous operations update the default values;
    g.fig.subplots_adjust(top=0.83,
                      bottom=0.15,
                      right=0.95,
                      left=0.05,
                      hspace=-0.20,
                      wspace=0.1)
    
    # Write the x-axis tick labels using percentages;
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{x:.1f}x"
    g.fig.get_axes()[-1].xaxis.set_major_formatter(major_formatter)
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    # Add custom legend;
    custom_lines = [Patch(facecolor=COLORS["peach1"], edgecolor="#2f2f2f", label="CUDA"),
                    Patch(facecolor=COLORS["b8"], edgecolor="#2f2f2f", label="GrCUDA"),
                    ]
    leg = g.fig.legend(custom_lines, ["CUDA", "GrCUDA"], bbox_to_anchor=(0.97, 0.98), fontsize=15)
    leg.set_title(None)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    # Main plot title;
    g.fig.suptitle("Relative Exec. Time Distribution,\nCUDA vs GrCUDA", ha="left", x=0.05, y=0.95, fontsize=18)
    
    return g

##############################
##############################

if __name__ == "__main__":
    data_grcuda = load_data(INPUT_DATE_GRCUDA, skip_iter=3)
    data_cuda = load_data_cuda(INPUT_DATE_CUDA, skip_iter=3)
    data = join_tables(data_grcuda, data_cuda)
    
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
            exec_time_axes += [build_exec_time_plot_grcuda_cuda(curr_res, gs, block_size_i, b_i)]
            
    plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=20, ha="center", va="center", xycoords="figure fraction")
    plt.annotate("Speedup", xy=(0.02, 0.5), fontsize=20, ha="center", va="center", rotation=90, xycoords="figure fraction")    
    plt.suptitle("Speedup of GrCUDA w.r.t. CUDA", fontsize=25, x=.05, y=0.99, ha="left")
    
    save_plot(PLOT_DIR, "speedup_baseline_grcuda_cuda_{}.{}", OUTPUT_DATE)

    
    #%% Similar plot, but all block sizes are on 1 row;
    
    sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    plt.rcParams["font.family"] = ["Latin Modern Roman"] 
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    # Lists of benchmarks and block sizes;
    benchmark_list = sorted(data["benchmark"].unique()) 
    policy_list = sorted(data["exec_policy"].unique())
    num_col = len(benchmark_list)
    num_row = len(policy_list)
    fig = plt.figure(figsize=(2.7 * num_col, 3.9 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.8,
                    bottom=0.14,
                    left=0.1,
                    right=0.95,
                    hspace=0.8,
                    wspace=0.15)
        
    exec_time_axes = []
    for b_i, b in enumerate(benchmark_list):
        for p_i, p in enumerate(policy_list): 
            curr_res = data[(data["benchmark"] == b) & (data["exec_policy"] == p)].reset_index(drop=True)  
            exec_time_axes += [build_exec_time_plot_grcuda_cuda_compact(curr_res, gs, p_i, b_i)]
        
    plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    plt.annotate("Speedup", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    plt.suptitle("Speedup of GrCUDA w.r.t. CUDA", fontsize=25, x=.05, y=0.99, ha="left")
    
    save_plot(PLOT_DIR, "speedup_baseline_grcuda_cuda_compact_{}.{}", OUTPUT_DATE)
    
    
    #%% Ridge plot with distributions;
    g = ridgeplot(data)
    save_plot(PLOT_DIR, "speedup_baseline_grcuda_cuda_ridgeplot_{}.{}", OUTPUT_DATE)