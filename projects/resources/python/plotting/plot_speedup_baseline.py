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
import math

import os
from load_data import load_data, compute_speedup
from plot_utils import COLORS, get_exp_label, get_ci_size, save_plot, remove_outliers_df_grouped


# INPUT_DATE = "2020_09_19_grcuda"
OUTPUT_DATE = "2020_10_14"
PLOT_DIR = "../../../../grcuda-data/plots"

BENCHMARK_NAMES = {"b1": "Vector Squares", "b5": "B&S", "b8": "Images", "b6": "ML Ensemble", "b7": "HITS", "b10": "DL"}

INPUT_DATE_960 = "960/2020_10_11_13_15_09_grcuda_baseline"
INPUT_DATE_P100 = "P100/2020_10_13_10_03_48_grcuda_baseline" # "2020_09_29_17_30_03_grcuda_forceprefetch"
# INPUT_DATE_P100_NP = "P100/2020_09_19_grcuda_no_prefetch"
INPUT_DATE_1660 = "1660/2020_10_13_18_21_04_grcuda_baseline"


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

    # Top y-lim is depends on the benchmark, and is multiple of 1.5;
    max_y_val = np.max(data.groupby(["block_size_str", "size_str"])["computation_speedup"].median())
    fixed_max_y_val = np.ceil(max_y_val / 1.5) * 1.5
    
    ax.set_ylim((0.8, fixed_max_y_val))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=45, ha="right", fontsize=9, rotation_mode="anchor")
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(8))
    ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=9)

    # if y == 0:
    #     ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=12)
    # else:
    #     ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
    #     # Hide tick markers;
    #     for tic in ax.yaxis.get_major_ticks():
    #         tic.tick1line.set_visible(False) 
    #         tic.tick2line.set_visible(False) 
    
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
    print(data.groupby(["size_str", "block_size_str"])["computation_speedup"].apply(gmean))
    data_averaged = data.groupby(["size_str", "block_size_str"], as_index=True)["computation_speedup"].apply(gmean).reset_index()
    order = data["block_size_str"].unique()
    ax = sns.scatterplot(x="size_str", y="computation_speedup", hue="block_size_str", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="block_size_str", hue_order=order, style_order=order, linewidth=0.05)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Top y-lim is depends on the benchmark, and is multiple of 1.5;
    max_y_val = np.max(data.groupby(["block_size_str", "size_str"])["computation_speedup"].median())
    fixed_max_y_val = np.ceil(max_y_val / 1.5) * 1.5
    
    ax.set_ylim((0.8, fixed_max_y_val))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=0, ha="center", fontsize=8)
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(8))
    ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=9)

    # if y == 0:
    #     ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=12)
    # else:
    #     ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
    #     # Hide tick markers;
    #     for tic in ax.yaxis.get_major_ticks():
    #         tic.tick1line.set_visible(False) 
    #         tic.tick2line.set_visible(False) 
    
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


def build_exec_time_plot_2_row(data, gridspec, fig, i, j):
    
    data["size_str"] = data["size"].astype(str)
    
    palette = [COLORS["peach1"], COLORS["b8"], COLORS["b2"], COLORS["b4"]]
    markers = ["o", "X", "D", "P"]
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[i, j])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)
    ax = sns.lineplot(x="size_str", y="computation_speedup", hue="block_size_str", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, zorder=2)
    print(data.groupby(["size_str", "block_size_str"])["computation_speedup"].apply(gmean))
    data_averaged = data.groupby(["size_str", "block_size_str"], as_index=True)["computation_speedup"].apply(gmean).reset_index()
    order = data["block_size_str"].unique()
    ax = sns.scatterplot(x="size_str", y="computation_speedup", hue="block_size_str", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="block_size_str", hue_order=order, style_order=order, linewidth=0.05)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    # Top y-lim is depends on the benchmark, and is multiple of 1.5;
    max_y_val = np.max(data.groupby(["block_size_str", "size_str"])["computation_speedup"].median())
    fixed_max_y_val = np.ceil(max_y_val / 1.5) * 1.5
    
    ax.set_ylim((0.9, fixed_max_y_val))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=0, ha="center", fontsize=9)
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(7))
    if j == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=10)
    else:
        ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
        # Hide tick markers;
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False) 
            tic.tick2line.set_visible(False) 
    
    ax.set_ylabel(None)     
    ax.set_xlabel(None) 
    
    # Add benchmark name and baseline execution time annotations;
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.05), fontsize=12, ha="center", xycoords="axes fraction")
    
     # Turn off tick lines;
    ax.xaxis.grid(False)
    
    # Add baseline execution time annotations (median of execution time across blocks);
    ax.annotate(f"Median baseline exec. time (ms):", xy=(0, -0.27), fontsize=9, ha="left", xycoords="axes fraction", color=COLORS["peach1"])
    for l_i, l in enumerate(labels):
        baseline_median = np.median(data[data["size"] == int(l)]["baseline_time_sec"])
        ax.annotate(f"{int(1000 * baseline_median)}", xy=(l_i,  -0.37), fontsize=9, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    # Legend; 
    if i == 0 and j == 0:
        legend_labels = [f"1D={x.split(',')[0]}" for x in data["block_size_str"].unique()]
        custom_lines = [
            lines.Line2D([], [], color="white", marker=markers[i], markersize=10, label=legend_labels[i], markerfacecolor=palette[i], markeredgecolor="#2f2f2f") 
            for i in range(len(legend_labels))]
        # Add fake entries to have a comment about 2d and 3d sizes;
        # custom_lines += [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)] * 2
        # legend_labels += ["", ""]
        # # Re-sort labels by transposing them;
        # custom_lines = np.array(custom_lines).reshape((-1, 2)).T.reshape(-1)
        # legend_labels = np.array(legend_labels).reshape((-1, 2)).T.reshape(-1)
        
        leg = fig.legend(custom_lines, legend_labels, 
                                 bbox_to_anchor=(0.99, 1), fontsize=10, ncol=len(legend_labels), handletextpad=0.1, columnspacing=0.2)
        leg.set_title("Block size:\n2D=8x8, 3D=4x4x4", prop={"size": 10})
        leg._legend_box.align = "left"
    
    return ax


def build_exec_time_plot_2_row_multigpu(data, gridspec, fig, i, j):
    
    data["size_str"] = data["size"].astype(str)
    
    # Add prefetching or not to GPU name;
    data["gpu_original"] = data["gpu"].copy()
    # data["gpu"] += np.where(data["exec_policy_full"] == "sync_f", ", sync with prefetch", "")
    
    palette = [COLORS["peach1"], COLORS["b8"], COLORS["b2"], COLORS["b3"], COLORS["b5"]][:len(data["gpu"].unique())]
    markers = ["o", "X", "D", "X", "D"][:len(data["gpu"].unique())]
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[i, j])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.1)
    ax = sns.lineplot(x="size_str", y="computation_speedup", hue="gpu", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, zorder=2)
    # print(data.groupby(["size_str", "gpu"])["computation_speedup"].apply(gmean))
    data_averaged = data.groupby(["size_str", "gpu"], as_index=True)["computation_speedup"].apply(gmean).reset_index()
    order = data["gpu"].unique()
    ax = sns.scatterplot(x="size_str", y="computation_speedup", hue="gpu", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="gpu", hue_order=order, style_order=order, linewidth=0.05)
    
    size_dict = {v: i for i, v in enumerate(sorted(data["size"].unique()))}
    
    # Top y-lim is depends on the benchmark, and is multiple of 1.5;
    max_y_val = np.max(data.groupby(["gpu", "size_str"])["computation_speedup"].median())
    # fixed_max_y_val = np.ceil(max_y_val / 1.5) * 1.5
    fixed_max_y_val = 3 if i == 0 else 1.8
    
    # Obtain max/min for each block size;
    max_speedup = {}
    min_speedup = {}
    data_block_aggregated = data.groupby(["size_str", "gpu", "block_size_str"], as_index=True)["computation_speedup"].apply(gmean).reset_index()
    for (size, gpu), g in data_block_aggregated.groupby(["size_str", "gpu"], as_index=True):
        curr_min = np.inf
        curr_min_b = 0
        curr_max = 0
        curr_max_b = 0
        for r_i, r in g.iterrows():
            if r["computation_speedup"] >= curr_max:
                curr_max = r["computation_speedup"]
                curr_max_b = r["block_size_str"]
            if r["computation_speedup"] <= curr_min:
                curr_min = r["computation_speedup"]
                curr_min_b = r["block_size_str"]
        if gpu not in max_speedup:
            max_speedup[gpu] = []
        if gpu not in min_speedup:
            min_speedup[gpu] = []
        max_speedup[gpu] += [(size, curr_max, curr_max_b)]
        min_speedup[gpu] += [(size, curr_min, curr_min_b)]
    for g in data["gpu"].unique():
        tmp_lines = [[(size_dict[int(e[0][0])], e[0][1]), (size_dict[int(e[0][0])], e[1][1])] for e in zip(min_speedup[g], max_speedup[g])]
        lc = LineCollection(tmp_lines, color="#888888", alpha=0.8, linewidths=0.5)
        ax.add_collection(lc)
    for g in data["gpu"].unique():
        for e in zip(min_speedup[g], max_speedup[g]):
            if (e[1][1] - e[0][1] > (0.3 if i == 0 else 0.1)) and not (b == "b6" and g in ["GTX960", "GTX1660 Super"]):
                v_offset = 0.05 if i == 0 else 0.01
                ax.annotate(f"{e[0][2].split(',')[0]}", xy=(size_dict[int(e[0][0])] + 0.02, e[0][1] - v_offset), fontsize=6, ha="left", va="center", color="#2f2f2f", alpha=0.9,)
                ax.annotate(f"{e[1][2].split(',')[0]}", xy=(size_dict[int(e[1][0])] + 0.02, min(fixed_max_y_val, e[1][1] + v_offset)), fontsize=6, ha="left", va="center", color="#2f2f2f", alpha=0.9,)

    labels = sorted(data["size"].unique())
    labels_str = [str(x) for x in labels]
    
    ax.set_ylim((0.8, fixed_max_y_val))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    odd_ticks = 0 if (len(labels_str) % 2 == 1) else 1
    ax.set_xticks([l for i, l in enumerate(labels_str) if i % 2 == odd_ticks])
    
    ax.set_xticklabels(labels=[get_exp_label(l) for i, l in enumerate(labels) if i % 2 == odd_ticks], rotation=0, ha="center", fontsize=9)
    ax.tick_params(labelcolor="black", pad=3)
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(8 if i == 0 else 6))
    if j == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=10)
    else:
        ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
        # Hide tick markers;
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False) 
            tic.tick2line.set_visible(False) 
    
    ax.set_ylabel(None)     
    ax.set_xlabel(None) 
    
    # Add benchmark name and baseline execution time annotations;
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.05), fontsize=12, ha="center", xycoords="axes fraction")
    
     # Turn off tick lines;
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
   
    # Add baseline execution time annotations (median of execution time across blocks);
    # curr_label_set = set([int(l) for l_i, l in enumerate(labels) if l_i % 2 == odd_ticks])
    # other_label_set = set([int(l) for l_i, l in enumerate(labels) if l_i % 2 != odd_ticks])
    gpus = ["960", "1660", "P100"]
    ax.annotate("Median baseline exec. time (ms):", xy=(0, -0.27), fontsize=9, ha="left", xycoords="axes fraction", color="#949494")
    for g_i, gpu in enumerate(data["gpu_original"].unique()):
        if g_i < len(gpus):
            if (j == 0):
                ax.annotate(f"{gpus[g_i]}:", xy=(-0.75, -0.37 - g_i * 0.1), fontsize=9, color=palette[g_i], ha="right", xycoords=("data", "axes fraction"))
                   
            # Always print the maximum number of ticks;
            # curr_sizes = set(data[data["gpu_original"] == gpu]["size"].unique())
            # odd_ticks_2 = odd_ticks if len(curr_sizes.intersection(curr_label_set)) > len(curr_sizes.intersection(other_label_set)) else int(not odd_ticks)
          
            for l_i, l in enumerate(labels):
                vals = data[(data["size"] == int(l)) & (data["gpu_original"] == gpu)]["baseline_time_sec"]
                baseline_median = np.median(vals) if len(vals) > 0 else np.nan
                # print(i, j, gpu, baseline_median)
                if not math.isnan(baseline_median) and l_i % 2 == odd_ticks:
                    ax.annotate(f"{int(1000 * baseline_median)}", xy=(l_i, -0.37 - g_i * 0.1), fontsize=9, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    # Legend; 
    if i == 0 and j == 0:
        legend_labels = data["gpu"].unique() # [f"1D={x.split(',')[0]}" for x in data["block_size_str"].unique()]
        custom_lines = [
            lines.Line2D([], [], color="white", marker=markers[i], markersize=10, label=legend_labels[i], markerfacecolor=palette[i], markeredgecolor="#2f2f2f") 
            for i in range(len(legend_labels))]
        # Add fake entries to have a comment about 2d and 3d sizes;
        # custom_lines += [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)] * 2
        # legend_labels += ["", ""]
        # # Re-sort labels by transposing them;
        # custom_lines = np.array(custom_lines).reshape((-1, 2)).T.reshape(-1)
        # legend_labels = np.array(legend_labels).reshape((-1, 2)).T.reshape(-1)
        
        leg = fig.legend(custom_lines, legend_labels, 
                                 bbox_to_anchor=(0.99, 1), fontsize=10, ncol=len(legend_labels), handletextpad=0.1, columnspacing=0.3)
        # leg.set_title("Block size:\n2D=8x8, 3D=4x4x4", prop={"size": 10})
        leg._legend_box.align = "left"
    
    return ax


    
#%%
if __name__ == "__main__":
    # data = load_data(INPUT_DATE, skip_iter=3)
    
    # # Ignore synchronous execution;
    # data = data[data["exec_policy"] != "sync"]
    
    # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
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
    # plt.subplots_adjust(top=0.8,
    #                 bottom=0.15,
    #                 left=0.2,
    #                 right=0.90,
    #                 hspace=1.1,
    #                 wspace=0.3)
        
    # exec_time_axes = []
    # for b_i, b in enumerate(benchmark_list):
    #     for block_size_i, block_size in enumerate(block_size_list): 
    #         curr_res = data[(data["benchmark"] == b) & (data["block_size_str"] == block_size)].reset_index(drop=True)  
    #         exec_time_axes += [build_exec_time_plot(curr_res, gs, block_size_i, b_i)]
            
    # plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=20, ha="center", va="center", xycoords="figure fraction")
    # plt.annotate("Speedup over serial scheduling", xy=(0.02, 0.5), fontsize=20, ha="center", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Execution time speedup\nover serial kernel scheduling", fontsize=25, x=.05, y=0.99, ha="left")
    
    # save_plot(PLOT_DIR, "speedup_baseline_{}.{}", OUTPUT_DATE)
    
    #%% Similar plot, but all block sizes are on 1 row;
    
    
    # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # plt.rcParams['axes.titlepad'] = 20 
    # plt.rcParams['axes.labelpad'] = 10 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    
    # # Lists of benchmarks and block sizes;
    # benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    # num_col = len(benchmark_list)
    # num_row = 1
    # fig = plt.figure(figsize=(2.6 * num_col, 4.1 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.65,
    #                 bottom=0.21,
    #                 left=0.1,
    #                 right=0.95,
    #                 hspace=1.1,
    #                 wspace=0.3)
        
    # exec_time_axes = []
    # for b_i, b in enumerate(benchmark_list):
    #     curr_res = data[data["benchmark"] == b].reset_index(drop=True)  
    #     exec_time_axes += [build_exec_time_plot_1_row(curr_res, gs, b_i)]
        
    # plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # plt.annotate("Speedup over\nserial scheduling", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Execution time speedup\nover serial kernel scheduling", fontsize=20, x=.05, y=0.92, ha="left")

    # save_plot(PLOT_DIR, "speedup_baseline_1_row_{}.{}", OUTPUT_DATE)
    
    
    #%% Similar plot, but formatted for 1-column on a paper;
    
    # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # plt.rcParams['axes.titlepad'] = 20 
    # plt.rcParams['axes.labelpad'] = 10 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    
    # data = data[~((data["benchmark"] == "b5") & (data["size"] == 3000000))]
    
    # # Lists of benchmarks and block sizes;
    # benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    # num_row = 2
    # num_col = len(benchmark_list) // num_row
    # fig = plt.figure(figsize=(2.2 * num_col, 2.7 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.82,
    #                 bottom=0.15,
    #                 left=0.08,
    #                 right=0.98,
    #                 hspace=0.55,
    #                 wspace=0.15)
        
    # exec_time_axes = []
    # speedups = []
    # for b_i, b in enumerate(benchmark_list):
    #     i = b_i // num_col
    #     j = b_i % num_col
    #     curr_res = data[data["benchmark"] == b].reset_index(drop=True)  
    #     curr_res = remove_outliers_df_grouped(curr_res, column="computation_speedup", group=["block_size_str", "size"])
    #     speedups += [curr_res.groupby(["size", "block_size_str"])["computation_speedup"].apply(gmean)]
    #     exec_time_axes += [build_exec_time_plot_2_row(curr_res, gs, fig, i, j)]
        
    # plt.annotate("Input number of elements", xy=(0.5, 0.02), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # # plt.annotate("Speedup over\nserial scheduling", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Parallel scheduler speedup\nover serial scheduler", fontsize=16, x=.02, y=0.99, ha="left")

    # save_plot(PLOT_DIR, "speedup_baseline_2_row_{}.{}", OUTPUT_DATE)
    
    #%% Plot both P100 and GTX960
    
    # data_960 = load_data(INPUT_DATE_960, skip_iter=3)
    # data_p100 = load_data(INPUT_DATE_P100, skip_iter=3)
    # data_1660 = load_data(INPUT_DATE_1660, skip_iter=3)
    # # data_p100_np = load_data(INPUT_DATE_P100_NP, skip_iter=3)
    # data_960["gpu"] = "GTX960"
    # data_p100["gpu"] = "P100"
    # data_1660["gpu"] = "GTX1660 Super"
    # # data_p100_np["gpu"] = "P100, no prefetch"
    # # data = pd.concat([data_960, data_p100, data_p100_np])
    # data = pd.concat([data_960, data_1660, data_p100]).reset_index(drop=True)
    
    # # data = data[data["force_prefetch"] == False]
    
    # # Ignore synchronous execution;
    # # data = data[data["exec_policy"] != "sync"]
    
    # # Remove no prefetch data if required;
    # # data = data[data["gpu"] != "P100, no prefetch"]
    
    # # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    # sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # plt.rcParams['axes.titlepad'] = 20 
    # plt.rcParams['axes.labelpad'] = 10 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    
    # # Lists of benchmarks and block sizes;
    # benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    # block_size_list = sorted(data["block_size_str"].unique(), key=lambda x: [int(y) for y in x.split(",")])
    
    # # Lists of benchmarks and block sizes;
    # benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    # num_row = 2
    # num_col = len(benchmark_list) // num_row
    # fig = plt.figure(figsize=(2.2 * num_col, 2.7 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.86,
    #                 bottom=0.18,
    #                 left=0.09,
    #                 right=0.98,
    #                 hspace=0.75,
    #                 wspace=0.1)
        
    # exec_time_axes = []
    # speedups = []
    # for b_i, b in enumerate(benchmark_list):
    #     i = b_i // num_col
    #     j = b_i % num_col
    #     curr_res = data[data["benchmark"] == b].reset_index(drop=True)  
    #     curr_res = remove_outliers_df_grouped(curr_res, column="computation_speedup", group=["block_size_str", "size", "gpu"])
    #     speedups += [curr_res.groupby(["size", "block_size_str", "gpu"])["computation_speedup"].apply(gmean)]
    #     exec_time_axes += [build_exec_time_plot_2_row_multigpu(curr_res, gs, fig, i, j)]
        
    # plt.annotate("Input number of elements", xy=(0.5, 0.02), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # # plt.annotate("Speedup over\nserial scheduling", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Parallel scheduler speedup\nover serial scheduler", fontsize=16, x=.02, y=0.99, ha="left")

    # save_plot(PLOT_DIR, "speedup_baseline_multigpu_{}.{}", OUTPUT_DATE)
    
    
    #%% Plot speedup with prefetching of sync and async w.r.t. sync baseline;
    
    data_960 = load_data(INPUT_DATE_960, skip_iter=3)
    data_p100 = load_data(INPUT_DATE_P100, skip_iter=3)
    data_1660 = load_data(INPUT_DATE_1660, skip_iter=3)
    data_960["gpu"] = "GTX960"
    data_p100["gpu"] = "P100"
    data_1660["gpu"] = "GTX1660 Super"
    data = pd.concat([data_960, data_1660, data_p100]).reset_index(drop=True)
    
    data["exec_policy_full"] = data["exec_policy"] + np.where(data["force_prefetch"], "_f", "")
    
    # Recompute speedups w.r.t. sync-noprefetch policy;
    compute_speedup(data, ["gpu", "benchmark", "new_stream_policy", "parent_stream_policy",
            "dependency_policy", "block_size_1d", "block_size_2d",
            "total_iterations", "cpu_validation", "random_init", "size", "realloc", "reinit"], baseline_filter_col="exec_policy_full", baseline_filter_val="sync")

    # Ignore synchronous execution;
    data = data[data["exec_policy_full"] != "sync"]
    # Skip no-prefetch;
    data = data[(data["exec_policy_full"] != ASYNC_POLICY_NAME) | (data["gpu"] == "GTX960")]
    data = data[(data["exec_policy_full"] != "sync_f") | (data["gpu"] == "GTX960")]
    
    # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    # Lists of benchmarks and block sizes;
    benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    block_size_list = sorted(data["block_size_str"].unique(), key=lambda x: [int(y) for y in x.split(",")])
    
    # Lists of benchmarks and block sizes;
    benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    num_row = 2
    num_col = len(benchmark_list) // num_row
    fig = plt.figure(figsize=(2.2 * num_col, 2.8 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.86,
                    bottom=0.18,
                    left=0.09,
                    right=0.98,
                    hspace=0.85,
                    wspace=0.1)
        
    exec_time_axes = []
    speedups = []
    for b_i, b in enumerate(benchmark_list):
        i = b_i // num_col
        j = b_i % num_col
        curr_res = data[data["benchmark"] == b].reset_index(drop=True)  
        curr_res = remove_outliers_df_grouped(curr_res, column="computation_speedup", group=["block_size_str", "size", "gpu"])
        speedups += [curr_res.groupby(["size", "block_size_str", "gpu"])["computation_speedup"].apply(gmean)]
        exec_time_axes += [build_exec_time_plot_2_row_multigpu(curr_res, gs, fig, i, j)]
        
    plt.annotate("Input number of elements (x-axis not to scale)", xy=(0.5, 0.02), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # plt.annotate("Speedup over\nserial scheduling", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    plt.suptitle("Parallel scheduler speedup\nover serial scheduler", fontsize=16, x=.02, y=0.99, ha="left")

    save_plot(PLOT_DIR, "speedup_baseline_multigpu_prefetch_{}.{}", OUTPUT_DATE)
    