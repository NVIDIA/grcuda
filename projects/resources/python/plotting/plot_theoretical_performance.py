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
Created on Mon Jun 29 12:00:01 2020

@author: alberto.parravicini
"""

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy.stats.mstats import gmean
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.lines as lines
import math

import os
from load_data import load_data, load_data_cuda, join_tables, compute_speedup
from segretini_matplottini.src.plot_utils import COLORS, get_exp_label, get_ci_size, save_plot, remove_outliers_df_grouped
import matplotlib.ticker as ticker

##############################
##############################

DEFAULT_RES_DIR = "../../../../grcuda-data/results/scheduling"

# INPUT_DATE_GRCUDA = "960/2020_09_29_20_11_01_grcuda_withphases_forceprefetch"
OUTPUT_DATE = "2020_10_14"
PLOT_DIR = "../../../../grcuda-data/plots"

INPUT_DATE_GRCUDA_P100 = "P100/2020_10_13_13_47_28_grcuda_with_phases"
INPUT_DATE_GRCUDA_960 = "960/2020_10_11_19_14_42_grcuda_with_phases"
INPUT_DATE_GRCUDA_1660 = "1660/2020_10_13_19_11_14_grcuda_with_phases"

B5_ITER = 10
B7_ITER = 5

BENCHMARK_NAMES = {"b1": "Vector Squares", "b5": "B&S", "b8": "Images", "b6": "ML Ensemble", "b7": "HITS", "b10": "DL"}
BENCHMARK_PHASES = {
    "b1": ["square_1", "square_2", "reduce"],
    "b5": [y for x in [[f"bs_{i}"] for i in range(B5_ITER)] for y in x],
    "b6": ["rr_1", "rr_2", "rr_3", "nb_1", "nb_2", "nb_3", "nb_4", "softmax_1", "softmax_2", "argmax"],
    "b7": [y for x in [[f"spmv_a_{i}", f"spmv_h_{i}", f"sum_a_{i}", f"sum_h_{i}", f"divide_a_{i}", f"divide_h_{i}", f"norm_reset_{i}"] for i in range(B7_ITER)] for y in x],
    "b8": ["blur_small", "blur_large", "blur_unsharpen", "sobel_small", "sobel_large", "maximum", "minimum", "extend", "unsharpen", "combine", "combine_2"],
    "b10": ["conv_x1", "pool_x1", "conv_x2", "conv_y1", "pool_y1", "conv_y2", "concat", "dot_product"],
    }

# ASYNC_POLICY_NAME = "async"   # If parsing new results;
ASYNC_POLICY_NAME = "default"  # If parsing older results;

##############################
##############################

def theoretical_speed(input_data, group_columns, benchmark):
    data = input_data.copy()
    
    # Only relevant for "sync" policy;
    data["theoretical_time_sec"] = THEORETICAL_SPEED_FUNCTIONS[benchmark](data)
    
    for key, group in data.groupby(group_columns):
        # Get the median theoretical time;
        median_theoretical_time = np.mean(group[group["exec_policy"] == "sync"]["theoretical_time_sec"])
        data.loc[group.index, "speedup_wrt_theoretical"] = median_theoretical_time / group["computation_sec"]
    return data

def theoretical_speed_b1(data):
    return np.maximum(data["square_1"], data["square_2"]) + data["reduce"]

def theoretical_speed_b5(data):
    return data[[f"bs_{i}" for i in range(B5_ITER)]].max(axis=1)

def theoretical_speed_b6(data):
    return np.maximum(data["rr_1"] + data["rr_2"] + data["rr_3"] + data["softmax_1"], data["nb_1"] + data["nb_2"] + data["nb_3"] + data["nb_4"] + data["softmax_2"]) + data["argmax"]

def theoretical_speed_b7(data):
    total = np.zeros(len(data))
    for i in range(B7_ITER):
        total += data[f"norm_reset_{i}"] + np.maximum(data[f"divide_a_{i}"] + np.maximum(data[f"spmv_a_{i}"] + data[f"sum_a_{i}"], data[f"spmv_h_{i}"]),
                                                      data[f"divide_h_{i}"] + np.maximum(data[f"spmv_h_{i}"] + data[f"sum_h_{i}"], data[f"spmv_a_{i}"]))
    return total

def theoretical_speed_b8(data):
    extend = np.maximum(data["maximum"], data["minimum"]) + data["extend"]
    combine = data["combine"] + np.maximum(data["blur_unsharpen"] + data["unsharpen"], data["blur_large"] + data["sobel_large"] + extend)
    return data["combine_2"] + np.maximum(data["blur_small"] + data["sobel_small"], combine)

def theoretical_speed_b10(data):
    return np.maximum(data["conv_x1"] + data["pool_x1"] + data["conv_x2"], data["conv_y1"] + data["pool_y1"] + data["conv_y2"]) + data["concat"] + data["dot_product"]

THEORETICAL_SPEED_FUNCTIONS = {
    "b1": theoretical_speed_b1,
    "b5": theoretical_speed_b5,
    "b6": theoretical_speed_b6,
    "b7": theoretical_speed_b7,
    "b8": theoretical_speed_b8,
    "b10": theoretical_speed_b10,
    }

##############################
##############################

def build_theoretical_time_plot(data, gridspec, x, y):
    
    palette = [COLORS["peach1"], COLORS["bb1"]]
    markers = ["o"] * len(palette)
    
    data["size_str"] = data["size"].astype(str)
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)

    ax = sns.lineplot(x="size_str", y="speedup_wrt_theoretical", hue="exec_policy", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, ci=None, sort=False, zorder=2)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Add rectangles to represent variance;
    for p_i, p in enumerate(sorted(data["exec_policy"].unique())):
        rectangles = []
        for s_i, s in enumerate(labels):
            curr_data = data[(data["size"] == s) & (data["exec_policy"] == p)]
            upper_ci_size, lower_ci_size, center = get_ci_size(curr_data["speedup_wrt_theoretical"], estimator=gmean, ci=0.90)
            bottom = center - lower_ci_size
            width = 0.1
            lower_left = [s_i - width / 2, bottom]
            # Add an offset to the x position, to avoid overlapping;
            lower_left[0] += (2 * p_i - 1) * (width / 3.5)
            rectangles += [Rectangle(lower_left, width, upper_ci_size + lower_ci_size)]
            
        pc = PatchCollection(rectangles, facecolor=palette[p_i], edgecolor="#2f2f2f", linewidth=0.5, zorder=3, clip_on=True, alpha=0.7)         
        ax.add_collection(pc)         

    # Set the same y limits in each plot;
    ax.set_ylim((0, 1))

    # Add a horizontal line to denote speedup = 1x;
    # ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=45, ha="right", fontsize=9, rotation_mode="anchor")
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(5))
    if y == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=9)
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
    ax.annotate(f"Min. theoretical time (ms):", xy=(0, -0.37), fontsize=9, ha="left", xycoords="axes fraction", color=COLORS["r4"])
    
    for i, l in enumerate(labels):
        baseline_median = np.median(data[(data["exec_policy"] == "sync") & (data["size"] == int(l))]["theoretical_time_sec"])
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


def build_theoretical_time_plot_compact(data, gridspec, x, y, baseline_labels=None):
    
    data["size_str"] = data["size"].astype(str)
    
    legend_labels = ["DAG Scheduling", "Serial Scheduling"]
    
    palette = [COLORS["peach1"], COLORS["b8"], COLORS["b2"], COLORS["b4"]][:len(data["block_size_str"].unique())]
    markers = ["o", "X", "D", "P"][:len(data["block_size_str"].unique())]
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)

    ax = sns.lineplot(x="size_str", y="speedup_wrt_theoretical", hue="block_size_str", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, zorder=2)
    data_averaged = data.groupby(["size_str", "block_size_str"], as_index=True)["speedup_wrt_theoretical"].apply(gmean).reset_index()
    order = data["block_size_str"].unique()
    ax = sns.scatterplot(x="size_str", y="speedup_wrt_theoretical", hue="block_size_str", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="block_size_str", hue_order=order, style_order=order, linewidth=0.05)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Set the same y limits in each plot;
    ax.set_ylim((0, 1))

    # Add a horizontal line to denote speedup = 1x;
    # ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=0, ha="center", fontsize=9)
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(5))
    if y == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=9)
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
    if baseline_labels:
        ax.annotate(f"Median baseline exec. time (ms):", xy=(0, -0.22), fontsize=9, ha="left", xycoords="axes fraction", color=COLORS["r4"])
        for i, l in enumerate(labels):
            baseline_median = baseline_labels[i]
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


def build_theoretical_time_plot_2rows(data, gridspec, x, y, baseline_labels=None):
    
    data["size_str"] = data["size"].astype(str)
    
    legend_labels = ["Serial Scheduler", "Parallel Scheduler"]
    
    palette = [COLORS["peach1"], COLORS["b8"], COLORS["b2"], COLORS["b4"]][:len(data["block_size_str"].unique())]
    markers = ["o", "X", "D", "P"][:len(data["block_size_str"].unique())]
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)

    ax = sns.lineplot(x="size_str", y="speedup_wrt_theoretical", hue="block_size_str", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, zorder=2)
    data_averaged = data.groupby(["size_str", "block_size_str"], as_index=True)["speedup_wrt_theoretical"].apply(gmean).reset_index()
    order = data["block_size_str"].unique()
    ax = sns.scatterplot(x="size_str", y="speedup_wrt_theoretical", hue="block_size_str", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="block_size_str", hue_order=order, style_order=order, linewidth=0.05)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Set the same y limits in each plot;
    ax.set_ylim((0, 1))

    # Add a horizontal line to denote speedup = 1x;
    # ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=0, ha="center", fontsize=8)
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(5))
    if y == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=9)
    else:
        ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
        # Hide tick markers;
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False) 
            tic.tick2line.set_visible(False) 
            
    # Add policy annotation;
    if y == 0 and x % 2 == 0:
        ax.annotate(f"{legend_labels[x // 2]}", xy=(-0.3, -1.4), fontsize=14, ha="center", xycoords="axes fraction", rotation=90) 
    
    ax.set_ylabel(None)     
    ax.set_xlabel(None) 
    
    # Add benchmark name and baseline execution time annotations;
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.1), fontsize=10, ha="center", xycoords="axes fraction")
    
     # Turn off tick lines;
    ax.xaxis.grid(False)
    
    # Add baseline execution time annotations (median of execution time across blocks);
    if baseline_labels:
        ax.annotate(f"Median baseline exec. time (ms):", xy=(0, -0.34), fontsize=8, ha="left", xycoords="axes fraction", color=COLORS["peach1"])
        for i, l in enumerate(labels):
            baseline_median = baseline_labels[i]
            ax.annotate(f"{int(1000 * baseline_median)}", xy=(i, -0.48), fontsize=8, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    # Legend; 
    if x == 0 and y== 0:
        legend_labels = [f"1D={x.split(',')[0]}" for x in data["block_size_str"].unique()]
        custom_lines = [
            lines.Line2D([], [], color="white", marker=markers[i], markersize=10, label=legend_labels[i], markerfacecolor=palette[i], markeredgecolor="#2f2f2f") 
            for i in range(len(legend_labels))]        
        leg = fig.legend(custom_lines, legend_labels, 
                                 bbox_to_anchor=(0.99, 1), fontsize=10, ncol=2, handletextpad=0.1, columnspacing=0.2)
        leg.set_title("Block size:\n2D=8x8, 3D=4x4x4", prop={"size": 10})
        leg._legend_box.align = "left"
    
    return ax


def build_theoretical_time_plot_2rows_default(data, gridspec, x, y, baseline_labels=None):
    
    data["size_str"] = data["size"].astype(str)
    
    legend_labels = ["Parallel Scheduler"]
    
    palette = [COLORS["peach1"], COLORS["b8"], COLORS["b2"], COLORS["b4"]][:len(data["block_size_str"].unique())]
    markers = ["o", "X", "D", "P"][:len(data["block_size_str"].unique())]
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)

    ax = sns.lineplot(x="size_str", y="speedup_wrt_theoretical", hue="block_size_str", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, zorder=2)
    data_averaged = data.groupby(["size_str", "block_size_str"], as_index=True)["speedup_wrt_theoretical"].apply(gmean).reset_index()
    order = data["block_size_str"].unique()
    ax = sns.scatterplot(x="size_str", y="speedup_wrt_theoretical", hue="block_size_str", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="block_size_str", hue_order=order, style_order=order, linewidth=0.05)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Set the same y limits in each plot;
    ax.set_ylim((0, 1.2))

    # Add a horizontal line to denote speedup = 1x;
    # ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=0, ha="center", fontsize=8)
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(7))
    if y == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=9)
    else:
        ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
        # Hide tick markers;
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False) 
            tic.tick2line.set_visible(False) 
            
    # Add policy annotation;
    if y == 0 and x % 2 == 0:
        ax.annotate(f"{legend_labels[x // 2]}", xy=(-0.3, -1.4), fontsize=14, ha="center", xycoords="axes fraction", rotation=90) 
    
    ax.set_ylabel(None)     
    ax.set_xlabel(None) 
    
    # Add benchmark name and baseline execution time annotations;
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.1), fontsize=10, ha="center", xycoords="axes fraction")
    
     # Turn off tick lines;
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    
    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
    
    # Add baseline execution time annotations (median of execution time across blocks);
    if baseline_labels:
        ax.annotate(f"Median baseline exec. time (ms):", xy=(0, -0.34), fontsize=8, ha="left", xycoords="axes fraction", color=COLORS["peach1"])
        for i, l in enumerate(labels):
            baseline_median = baseline_labels[i]
            ax.annotate(f"{int(1000 * baseline_median)}", xy=(i, -0.48), fontsize=8, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    # Legend; 
    if x == 0 and y== 0:
        legend_labels = [f"1D={x.split(',')[0]}" for x in data["block_size_str"].unique()]
        custom_lines = [
            # Patch(facecolor="white", marker=markers[i], markersize=10, label=legend_labels[i], markerfacecolor=palette[i], markeredgecolor="#2f2f2f") 
            lines.Line2D([0], [0], linestyle="none", marker=markers[i], markersize=10, label=legend_labels[i], markerfacecolor=palette[i], markeredgecolor="#2f2f2f") 
            for i in range(len(legend_labels))]        
        leg = fig.legend(custom_lines, legend_labels, 
                                 bbox_to_anchor=(0.99, 1), fontsize=10, ncol=2, handletextpad=0.1, columnspacing=0.2)
        leg.set_title("Block size:\n2D=8x8, 3D=4x4x4", prop={"size": 10})
        leg._legend_box.align = "left"
    
    return ax


def build_theoretical_time_plot_2rows_multigpu(data, gridspec, x, y, baseline_labels=None):
    
    data["size_str"] = data["size"].astype(str)
    
    legend_labels = ["Parallel Scheduler"]
        
    palette = [COLORS["peach1"], COLORS["b8"], COLORS["b2"], COLORS["b4"]][:len(data["gpu"].unique())]
    markers = ["o", "X", "D", "P"][:len(data["gpu"].unique())]
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)

    ax = sns.lineplot(x="size_str", y="speedup_wrt_theoretical", hue="gpu", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, zorder=2)
    data_averaged = data.groupby(["size_str", "gpu"], as_index=True)["speedup_wrt_theoretical"].apply(gmean).reset_index()
    order = data["gpu"].unique()
    ax = sns.scatterplot(x="size_str", y="speedup_wrt_theoretical", hue="gpu", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="gpu", hue_order=order, style_order=order, linewidth=0.05)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Set the same y limits in each plot;
    ax.set_ylim((0, 1.2))

    # Add a horizontal line to denote speedup = 1x;
    # ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    # ax.set_xticks(labels_str)
    # ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=0, ha="center", fontsize=8)
    # ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(7))
    # if y == 0:
    #     ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=9)
    # else:
    #     ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
    #     # Hide tick markers;
    #     for tic in ax.yaxis.get_major_ticks():
    #         tic.tick1line.set_visible(False) 
    #         tic.tick2line.set_visible(False) 
            
    # Set the x ticks;
    odd_ticks = 0 if (len(labels_str) % 2 == 1) else 1
    ax.set_xticks([l for i, l in enumerate(labels_str) if i % 2 == odd_ticks])
    
    ax.set_xticklabels(labels=[get_exp_label(l) for i, l in enumerate(labels) if i % 2 == odd_ticks], rotation=0, ha="center", fontsize=9)
    ax.tick_params(labelcolor="black", pad=3)
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(7))
    if y == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=10)
    else:
        ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
        # Hide tick markers;
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False) 
            tic.tick2line.set_visible(False) 
            
    # Add policy annotation;
    if y == 0 and x % 2 == 0:
        ax.annotate(f"{legend_labels[x // 2]}", xy=(-0.3, -1.4), fontsize=14, ha="center", xycoords="axes fraction", rotation=90) 
    
    ax.set_ylabel(None)     
    ax.set_xlabel(None) 
    
    # Add benchmark name and baseline execution time annotations;
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.1), fontsize=10, ha="center", xycoords="axes fraction")
    
     # Turn off tick lines;
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    
    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
    
    # # Add baseline execution time annotations (median of execution time across blocks);
    # if baseline_labels:
    #     ax.annotate(f"Median baseline exec. time (ms):", xy=(0, -0.34), fontsize=8, ha="left", xycoords="axes fraction", color=COLORS["peach1"])
    #     for i, l in enumerate(labels):
    #         baseline_median = baseline_labels[i]
    #         ax.annotate(f"{int(1000 * baseline_median)}", xy=(i, -0.48), fontsize=8, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    ax.annotate("Contention-free exec. time (ms):", xy=(0, -0.35), fontsize=9, ha="left", xycoords="axes fraction", color="#949494")
    gpus = ["960", "1660", "P100"]
    for g_i, gpu in enumerate(data["gpu"].unique()):
        if g_i < len(gpus):
            if (j == 0):
                ax.annotate(f"{gpus[g_i]}:", xy=(-0.75, -0.47 - g_i * 0.1), fontsize=9, color=palette[g_i], ha="right", xycoords=("data", "axes fraction"))
                       
            for l_i, l in enumerate(labels):
                try:
                    baseline_median = baseline_labels[gpu][data["benchmark"].unique()[0]][l]
                except KeyError:
                    baseline_median = np.nan
                # print(i, j, gpu, baseline_median)
                if not math.isnan(baseline_median) and l_i % 2 == odd_ticks:
                    ax.annotate(f"{int(1000 * baseline_median)}", xy=(l_i, -0.47 - g_i * 0.1), fontsize=9, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))

    # Legend; 
    if x == 0 and y== 0:
        legend_labels = data["gpu"].unique()
        custom_lines = [
            # Patch(facecolor="white", marker=markers[i], markersize=10, label=legend_labels[i], markerfacecolor=palette[i], markeredgecolor="#2f2f2f") 
            lines.Line2D([0], [0], linestyle="none", marker=markers[i], markersize=10, label=legend_labels[i], markerfacecolor=palette[i], markeredgecolor="#2f2f2f") 
            for i in range(len(legend_labels))]        
        leg = fig.legend(custom_lines, legend_labels, 
                                  bbox_to_anchor=(0.99, 1), fontsize=10, ncol=1, handletextpad=0.1, columnspacing=0.2)
        leg.set_title(None)
        leg._legend_box.align = "left"
    
    return ax


##############################
##############################

#%%

if __name__ == "__main__":
    
    # Columns that uniquely identify each benchmark setup;
    index_columns = ["benchmark", "exec_policy",
                     # "new_stream_policy", "parent_stream_policy", "dependency_policy",
                     "block_size_1d", "block_size_2d",
                     # "total_iterations", "cpu_validation", "random_init", 
                     "size",
                     # "realloc", "reinit"
                     ]
    
    processed_data_p100 = []
    processed_data_summary_p100 = []
    for b in BENCHMARK_PHASES.keys():
        data_b = load_data(INPUT_DATE_GRCUDA_P100, skip_iter=3, benchmark=b, phases=BENCHMARK_PHASES[b])
        data_b = data_b[data_b["force_prefetch"] == True]
        if len(data_b) > 0:
            data_b = remove_outliers_df_grouped(data_b, column="computation_speedup", group=["exec_policy", "benchmark", "block_size_1d", "block_size_2d", "size", "force_prefetch"]).reset_index(drop=True)
            tmp_cols = index_columns.copy()
            tmp_cols.remove("exec_policy")
            data_b = theoretical_speed(data_b, tmp_cols, b)
            data_b["gpu"] = "P100"
            data_summary = data_b.groupby(index_columns + ["gpu"])[["computation_speedup", "speedup_wrt_theoretical"]].aggregate(gmean).reset_index()
            processed_data_p100 += [data_b[list(data_b.columns[:20]) + list(data_b.columns[-4:])]]
            processed_data_summary_p100 += [data_summary]
            
    processed_data_960 = []
    processed_data_summary_960 = []
    for b in BENCHMARK_PHASES.keys():
        data_b = load_data(INPUT_DATE_GRCUDA_960, skip_iter=3, benchmark=b, phases=BENCHMARK_PHASES[b])
        if len(data_b) > 0:
            data_b = remove_outliers_df_grouped(data_b, column="computation_speedup", group=["exec_policy", "benchmark", "block_size_1d", "block_size_2d", "size", "force_prefetch"]).reset_index(drop=True)
            tmp_cols = index_columns.copy()
            tmp_cols.remove("exec_policy")
            data_b = theoretical_speed(data_b, tmp_cols, b)
            data_b["gpu"] = "GTX960"
            data_summary = data_b.groupby(index_columns + ["gpu"])[["computation_speedup", "speedup_wrt_theoretical"]].aggregate(gmean).reset_index() 
            processed_data_960 += [data_b[list(data_b.columns[:20]) + list(data_b.columns[-4:])]]
            processed_data_summary_960 += [data_summary]
        
    processed_data_1660 = []
    processed_data_summary_1660 = []
    for b in BENCHMARK_PHASES.keys():
        data_b = load_data(INPUT_DATE_GRCUDA_1660, skip_iter=3, benchmark=b, phases=BENCHMARK_PHASES[b])
        data_b = data_b[data_b["force_prefetch"] == True]
        if len(data_b) > 0:
            data_b = remove_outliers_df_grouped(data_b, column="computation_speedup", group=["exec_policy", "benchmark", "block_size_1d", "block_size_2d", "size", "force_prefetch"]).reset_index(drop=True)
            tmp_cols = index_columns.copy()
            tmp_cols.remove("exec_policy")
            data_b = theoretical_speed(data_b, tmp_cols, b)
            data_b["gpu"] = "GTX1660 Super"
            data_summary = data_b.groupby(index_columns + ["gpu"])[["computation_speedup", "speedup_wrt_theoretical"]].aggregate(gmean).reset_index() 
            processed_data_1660 += [data_b[list(data_b.columns[:20]) + list(data_b.columns[-4:])]]
            processed_data_summary_1660 += [data_summary]    
        
    data = pd.concat(processed_data_960 + processed_data_1660 + processed_data_p100).reset_index(drop=True)
    
    #%%
        
    # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    # plt.rcParams["font.family"] = ["Latin Modern Roman"]
    # plt.rcParams['axes.titlepad'] = 20 
    # plt.rcParams['axes.labelpad'] = 10 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    
    # # Lists of benchmarks and block sizes;
    # benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    # block_size_list = sorted(data["block_size_str"].unique(), key=lambda x: [int(y) for y in x.split(",")])
    # num_col = len(benchmark_list)
    # num_row = len(block_size_list)
    # fig = plt.figure(figsize=(2.5 * num_col, 4 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.85,
    #                 bottom=0.15,
    #                 left=0.2,
    #                 right=0.90,
    #                 hspace=1.2,
    #                 wspace=0.15)
        
    # exec_time_axes = []
    # for b_i, b in enumerate(benchmark_list):
    #     for block_size_i, block_size in enumerate(block_size_list): 
    #         curr_res = data[(data["benchmark"] == b) & (data["block_size_str"] == block_size)].reset_index(drop=True)  
    #         exec_time_axes += [build_theoretical_time_plot(curr_res, gs, block_size_i, b_i)]
            
    # plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=20, ha="center", va="center", xycoords="figure fraction")
    # plt.annotate("Speedup", xy=(0.02, 0.5), fontsize=20, ha="center", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Speedup w.r.t\nminimum theoretical time", fontsize=25, x=.05, y=0.99, ha="left")
    
    # save_plot(PLOT_DIR, "speedup_theoretical_time_{}.{}", OUTPUT_DATE)

    
    #%% Similar plot, but all block sizes are on 1 row;
    
    # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    # plt.rcParams["font.family"] = ["Latin Modern Roman"] 
    # plt.rcParams['axes.titlepad'] = 20 
    # plt.rcParams['axes.labelpad'] = 10 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    
    # # Lists of benchmarks and block sizes;
    # benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    # policy_list = sorted(data["exec_policy"].unique())
    # num_col = len(benchmark_list)
    # num_row = len(policy_list)
    # fig = plt.figure(figsize=(2.7 * num_col, 3.9 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.8,
    #                 bottom=0.14,
    #                 left=0.1,
    #                 right=0.95,
    #                 hspace=0.8,
    #                 wspace=0.15)
        
    # exec_time_axes = []
    # for b_i, b in enumerate(benchmark_list):
    #     baselines = []
    #     tmp_data = data[(data["exec_policy"] == "sync") & (data["benchmark"] == b)]
    #     labels = sorted(tmp_data["size"].unique())
    #     for i, l in enumerate(labels):
    #         baselines += [np.median(tmp_data[tmp_data["size"] == int(l)]["theoretical_time_sec"])]
    #     for p_i, p in enumerate(policy_list): 
    #         curr_res = data[(data["benchmark"] == b) & (data["exec_policy"] == p)].reset_index(drop=True)  
    #         exec_time_axes += [build_theoretical_time_plot_compact(curr_res, gs, p_i, b_i, baseline_labels=baselines)]
        
    # plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # plt.annotate("Speedup", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Speedup w.r.t\nminimum theoretical time", fontsize=25, x=.05, y=0.99, ha="left")
    
    # save_plot(PLOT_DIR, "speedup_theoretical_time_compact_{}.{}", OUTPUT_DATE)
    
    #%% Similar plot, but formatted for 1-column on a paper;
    
    # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # plt.rcParams['axes.titlepad'] = 20 
    # plt.rcParams['axes.labelpad'] = 10 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    # plt.rcParams['xtick.major.pad'] = 4
    
    # # Lists of benchmarks and block sizes;
    # benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    # policy_list = list(reversed(sorted(data["exec_policy"].unique())))
    # num_col = len(benchmark_list) // 2
    # num_row = len(policy_list) * 2
    # fig = plt.figure(figsize=(2.2 * num_col, 2 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.85,
    #                 bottom=0.10,
    #                 left=0.10,
    #                 right=0.98,
    #                 hspace=0.9,
    #                 wspace=0.15)
        
    # exec_time_axes = []
    # baselines_dict = {}
    # for b_i, b in enumerate(benchmark_list):
    #     baselines = []
    #     tmp_data = data[(data["exec_policy"] == "sync") & (data["benchmark"] == b)]
    #     labels = sorted(tmp_data["size"].unique())
    #     for i, l in enumerate(labels):
    #         baselines += [np.median(tmp_data[tmp_data["size"] == int(l)]["theoretical_time_sec"])]
    #     baselines_dict[b] = baselines
        
    # for p_i, p in enumerate(policy_list): 
    #     for b_i, b in enumerate(benchmark_list):
    #         index_tot = (len(benchmark_list) * p_i + b_i)
    #         j = index_tot % num_col
    #         i = index_tot // num_col
    #         curr_res = data[(data["benchmark"] == b) & (data["exec_policy"] == p)].reset_index(drop=True)  
    #         exec_time_axes += [build_theoretical_time_plot_2rows(curr_res, gs, i, j, baseline_labels=baselines_dict[b])]
        
    # plt.annotate("Input number of elements", xy=(0.5, 0.02), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # plt.suptitle("Slowdown w.r.t. execution\nwithout resource contention", fontsize=16, x=.02, y=0.99, ha="left")
    
    # l1 = lines.Line2D([0.01, 0.99], [0.46, 0.46], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--", linewidth=1)
    # fig.lines.extend([l1])

    # save_plot(PLOT_DIR, "speedup_theoretical_time_2rows_{}.{}", OUTPUT_DATE)
        
    #%% Similar plot, but formatted for 1-column on a paper and without serial execution time;
    
    # data = pd.concat(processed_data_1660).reset_index(drop=True)
    
    # # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    # sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})

    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # plt.rcParams['axes.titlepad'] = 20 
    # plt.rcParams['axes.labelpad'] = 10 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    # plt.rcParams['xtick.major.pad'] = 4
    
    # # Lists of benchmarks and block sizes;
    # benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    # policy_list = list(reversed(sorted(data["exec_policy"].unique())))
    # num_col = len(benchmark_list) // 2
    # num_row = len(policy_list)
    # fig = plt.figure(figsize=(2.2 * num_col, 2.4 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.75,
    #                 bottom=0.18,
    #                 left=0.10,
    #                 right=0.98,
    #                 hspace=0.9,
    #                 wspace=0.15)
        
    # exec_time_axes = []
    # baselines_dict = {}
    # for b_i, b in enumerate(benchmark_list):
    #     baselines = []
    #     tmp_data = data[(data["exec_policy"] == "sync") & (data["benchmark"] == b)]
    #     labels = sorted(tmp_data["size"].unique())
    #     for i, l in enumerate(labels):
    #         baselines += [np.median(tmp_data[tmp_data["size"] == int(l)]["theoretical_time_sec"])]
    #     baselines_dict[b] = baselines
    
    # policy_list = [ASYNC_POLICY_NAME]  # Skip sync policy;
    # for p_i, p in enumerate(policy_list): 
    #     for b_i, b in enumerate(benchmark_list):
    #         index_tot = (len(benchmark_list) * p_i + b_i)
    #         j = index_tot % num_col
    #         i = index_tot // num_col
    #         curr_res = data[(data["benchmark"] == b) & (data["exec_policy"] == p)].reset_index(drop=True)  
    #         exec_time_axes += [build_theoretical_time_plot_2rows_default(curr_res, gs, i, j, baseline_labels=baselines_dict[b])]
        
    # plt.annotate("Input number of elements", xy=(0.5, 0.02), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # plt.suptitle("Slowdown with respect to execution\nwithout resource contention,\nGTX 1660 Super", fontsize=16, x=.02, y=0.99, ha="left")
    
    # # l1 = lines.Line2D([0.01, 0.99], [0.46, 0.46], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--", linewidth=1)
    # # fig.lines.extend([l1])

    # save_plot(PLOT_DIR, "speedup_theoretical_time_2rows_default_{}.{}", OUTPUT_DATE)
    
    #%% Similar plot, but using multiple GPUs
    
    # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})

    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    plt.rcParams['xtick.major.pad'] = 4
    
    # Lists of benchmarks and block sizes;
    benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    policy_list = list(reversed(sorted(data["exec_policy"].unique())))
    num_col = len(benchmark_list) // 2
    num_row = len(policy_list)
    fig = plt.figure(figsize=(2.2 * num_col, 2.5 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.79,
                    bottom=0.18,
                    left=0.10,
                    right=0.98,
                    hspace=1,
                    wspace=0.15)
        
    exec_time_axes = []
    baselines_dict = {}
    for g in data["gpu"].unique():
        baselines_dict[g] = {}
        for b_i, b in enumerate(benchmark_list):
            baselines = {}
            tmp_data = data[(data["exec_policy"] == "sync") & (data["benchmark"] == b) & (data["gpu"] == g)]
            labels = sorted(tmp_data["size"].unique())
            for i, l in enumerate(labels):
                baselines[l] = np.median(tmp_data[tmp_data["size"] == int(l)]["theoretical_time_sec"])
            baselines_dict[g][b] = baselines
    
    policy_list = [ASYNC_POLICY_NAME]  # Skip sync policy;
    for p_i, p in enumerate(policy_list): 
        for b_i, b in enumerate(benchmark_list):
            index_tot = (len(benchmark_list) * p_i + b_i)
            j = index_tot % num_col
            i = index_tot // num_col
            curr_res = data[(data["benchmark"] == b) & (data["exec_policy"] == p)].reset_index(drop=True)  
            exec_time_axes += [build_theoretical_time_plot_2rows_multigpu(curr_res, gs, i, j, baseline_labels=baselines_dict)]
        
    plt.annotate("Input number of elements (x-axis not to scale)", xy=(0.5, 0.02), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    plt.suptitle("Slowdown with respect to execution\nwithout resource contention", fontsize=16, x=.02, y=0.99, ha="left")
    
    # l1 = lines.Line2D([0.01, 0.99], [0.46, 0.46], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--", linewidth=1)
    # fig.lines.extend([l1])

    save_plot(PLOT_DIR, "speedup_theoretical_time_2rows_multigpu_{}.{}", OUTPUT_DATE)
    
    #%%
    
    # slowdown = gmean(data["grcuda_cuda_speedup"])
    # print(slowdown)
    
    # slowdown_dict = {"sync": [], ASYNC_POLICY_NAME: []}
    # for i, g in data.groupby(["benchmark", "exec_policy"]):
    #     max_size = g["size"].max()
    #     slowdown_dict[i[1]] += [gmean(g[g["size"] == max_size]["grcuda_cuda_speedup"])]
    # print(gmean(slowdown_dict["sync"]), gmean(slowdown_dict[ASYNC_POLICY_NAME]))