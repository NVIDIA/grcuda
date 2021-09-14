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
import math

import os
from load_data import load_data, load_data_cuda, join_tables, join_tables_baseline
from plot_utils import COLORS, get_exp_label, get_ci_size, save_plot, update_width, add_labels, get_upper_ci_size, remove_outliers_df_grouped
import matplotlib.ticker as ticker

##############################
##############################

# # P100
# INPUT_DATE_GRCUDA = "2020_09_19_2_grcuda"
# INPUT_DATE_CUDA = "2020_09_25_09_29_10_cuda"
# # 960
# INPUT_DATE_GRCUDA = "2020_09_22_17_44_41_grcuda_b8baseline"
# INPUT_DATE_CUDA = "2020_09_22_18_36_21_cuda"

OUTPUT_DATE = "2020_10_14"
PLOT_DIR = "../../../../grcuda-data/plots"

# INPUT_DATE_GRCUDA_P100 = "2020_09_19_2_grcuda"
# INPUT_DATE_GRCUDA_960 = "2020_09_22_17_44_41_grcuda_b8baseline"
INPUT_DATE_CUDA_P100 = "P100/2020_10_13_10_04_06_cuda" # "P100/2020_09_25_09_29_10_cuda" 
INPUT_DATE_CUDA_960 = "960/2020_10_07_17_08_41_cuda"
INPUT_DATE_CUDA_1660 = "1660/2020_10_13_14_49_29_cuda"

INPUT_DATE_GRCUDA_960 = "960/2020_10_11_13_15_09_grcuda_baseline"
INPUT_DATE_GRCUDA_P100 = "P100/2020_10_13_10_03_48_grcuda_baseline" # "P100/2020_10_06_grcuda_p100_baseline" # "2020_09_29_17_30_03_grcuda_forceprefetch"
INPUT_DATE_GRCUDA_1660 = "1660/2020_10_13_18_21_04_grcuda_baseline"

BENCHMARK_NAMES = {"b1": "Vector Squares", "b5": "B&S", "b8": "Images", "b6": "ML Ensemble", "b7": "HITS", "b10": "DL"}

# ASYNC_POLICY_NAME = "async"   # If parsing new results;
ASYNC_POLICY_NAME = "default"  # If parsing older results;

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
    odd_ticks = 0 if (len(labels_str) % 2 == 1) else 1
    ax.set_xticks([l for i, l in enumerate(labels_str) if i % 2 == odd_ticks])
    ax.set_xticklabels(labels=[get_exp_label(l) for i, l in enumerate(labels) if i % 2 == odd_ticks], rotation=0, ha="center", fontsize=9)
    ax.tick_params(labelcolor="black", pad=3)
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
                
    # Set the x ticks;
    # ax.set_xticks(labels_str)
    # ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=45, ha="right", fontsize=9, rotation_mode="anchor")
    # ax.tick_params(labelcolor="black")
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
        legend_labels = ["Paraller Scheduler", "Serial Scheduler"]
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
    
    legend_labels = ["Paraller Scheduler", "Serial Scheduler"]
    
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
        ax.annotate(f"{legend_labels[x % 2]}", xy=(-0.15, 1.25), fontsize=14, ha="left", xycoords="axes fraction") 
    
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


def build_exec_time_plot_grcuda_cuda_2rows(data, gridspec, x, y):
        
    data["size_str"] = data["size"].astype(str)
    
    legend_labels = ["Paraller Scheduler", "Serial Scheduler"]
    
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
    ax.set_ylim((0.0, 1.5))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Set the x ticks;
    ax.set_xticks(labels_str)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=0, ha="center", fontsize=8)
    ax.tick_params(labelcolor="black")
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(4))
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
    ax.annotate(f"Median baseline exec. time (ms):", xy=(0, -0.42), fontsize=8, ha="left", xycoords="axes fraction", color=COLORS["peach1"])
    for i, l in enumerate(labels):
        baseline_median = np.median(data[data["size"] == int(l)]["baseline_time_sec_cuda"])
        ax.annotate(f"{int(1000 * baseline_median)}", xy=(i,  -0.57), fontsize=8, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
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


def build_exec_time_plot_grcuda_cuda_2rows_multigpu(data, gridspec, x, y, exec_policy, palette_in, markers_in):
    
    # data = pd.melt(data, id_vars=["benchmark", "size", "block_size_str", "computation_sec"], value_vars=data.columns[-3:],
    #     var_name="versus", value_name="speedup")
    # data["size_str"] = data["size"].astype(str)
    
    if exec_policy == ASYNC_POLICY_NAME:
        data = data[~data["versus"].isin(["speedup_sync", "speedup_cudagraphsingle"])]
        if len(data) == 0:
            return
    elif exec_policy == "sync":
        data = data[data["versus"].isin(["speedup_cudagraphsingle"])]
        if len(data) == 0:
            return
    else:
        raise ValueError(exec_policy + " is not ok!")
    # data = data[data["versus"] != "speedup_sync"]
    # print(x,y,exec_policy,len(data))
    legend_labels = ["Serial Scheduler", "Paraller Scheduler"]
    
    order = data["versus"].unique()
    palette = [palette_in[o] for o in order]
    markers = [markers_in[o] for o in order]
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.3)

    ax = sns.lineplot(x="size_str", y="speedup", hue="versus", data=data[data["gpu"] == "GTX960"], palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, hue_order=order, zorder=2)
    ax = sns.lineplot(x="size_str", y="speedup", hue="versus", data=data[data["gpu"] == "P100"], palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, hue_order=order, zorder=2)
    data_averaged = data.groupby(["size_str", "versus", "gpu"], as_index=True)["speedup"].apply(gmean).reset_index()
    
    ax = sns.scatterplot(x="size_str", y="speedup", hue="versus", data=data_averaged[data_averaged["gpu"] == "GTX960"], palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="versus", hue_order=order, style_order=order, linewidth=0.05)
    ax = sns.scatterplot(x="size_str", y="speedup", hue="versus", data=data_averaged[data_averaged["gpu"] == "P100"], palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="versus", hue_order=order, style_order=order, linewidth=0.05)
    
    labels = sorted(data["size"].unique())
    labels_str = [str(l) for l in labels]
    
    # Set the same y limits in each plot;
    num_y_ticks = 6
    if exec_policy == "sync":
        ax.set_ylim((0.5, 1.5))
        num_y_ticks = 5
    elif exec_policy == ASYNC_POLICY_NAME and x == 3:
        ax.set_ylim((0.5, 2.0))
        num_y_ticks = 4
    else:
        ax.set_ylim((0.5, 3))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Add a vertical line to split GPUs;
    max_size_960 = str(data[data["gpu"] == "GTX960"]["size"].max())
    
    # Set the x ticks;
    odd_ticks = 0 if (len(labels_str) % 2 == 1) else 1
    xticks = []
    max_tick_960 = 0
    for i, l in enumerate(labels_str):
        if i % 2 == odd_ticks:
            xticks += [l]
        if l == max_size_960:
            max_tick_960 = i
    ax.axvline(x=max_tick_960, color="#2f2f2f", linestyle="--", zorder=1, linewidth=0.5, alpha=0.5)
    ax.annotate("GTX960", xy=(0.35, 0.85), fontsize=8, ha="center", xycoords="axes fraction", color="#2f2f2f", alpha=0.5)
    ax.annotate("P100", xy=(0.6, 0.85), fontsize=8, ha="center", xycoords="axes fraction", color="#2f2f2f", alpha=0.5)        
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels=[get_exp_label(l) for i, l in enumerate(labels) if i % 2 == odd_ticks], rotation=0, ha="center", fontsize=9)
    ax.tick_params(labelcolor="black", pad=3)

    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(num_y_ticks))
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
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.08), fontsize=10, ha="center", xycoords="axes fraction")
    
    # Turn off tick lines;
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    # ax.tick_params(axis="x", which="major",length=3)
    
    # Add baseline execution time annotations (median of execution time across blocks);
    gpus = ["960", "P100"]
    palette_gpu = [COLORS["peach1"], COLORS["b8"], COLORS["b2"]]
    ax.annotate("Median GrCUDA exec. time (ms):", xy=(0, -0.42), fontsize=9, ha="left", xycoords="axes fraction", color="#949494")
    for g_i, gpu in enumerate(data["gpu"].unique()):
        if g_i < len(gpus):
            if (j == 0):
                ax.annotate(f"{gpus[g_i]}:", xy=(-0.75, -0.57 - g_i * 0.15), fontsize=9, color=palette_gpu[g_i], ha="right", xycoords=("data", "axes fraction"))
            for l_i, l in enumerate(labels):
                vals = data[(data["size"] == int(l)) & (data["gpu"] == gpu)]["computation_sec"]
                baseline_median = np.median(vals) if len(vals) > 0 else np.nan
                if not math.isnan(baseline_median) and l_i % 2 == odd_ticks:
                    ax.annotate(f"{int(1000 * baseline_median)}", xy=(l_i, -0.57 - g_i * 0.15), fontsize=9, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    
    # # Add baseline execution time annotations (median of execution time across blocks);
    # ax.annotate("Median GrCUDA exec. time (ms):", xy=(0, -0.42), fontsize=8, ha="left", xycoords="axes fraction", color=COLORS["peach1"])
    # for i, l in enumerate(labels):    
    #     baseline_median = np.median(data[data["size"] == int(l)]["computation_sec"])
    #     ax.annotate(f"{int(1000 * baseline_median)}", xy=(i, -0.57 - j * 0.1), fontsize=8, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    return ax


def build_exec_time_plot_grcuda_cuda_3rows_multigpu(data, gridspec, x, y, gpu, palette_in, markers_in, sizes=None):
    
    legend_label = "Paraller Scheduler"
    
    order = data["versus"].unique()
    palette = [palette_in[o] for o in order]
    markers = [markers_in[o] for o in order]
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.3)

    ax = sns.lineplot(x="size_str", y="speedup", hue="versus", data=data, palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, hue_order=order, zorder=2)
    data_averaged = data.groupby(["size_str", "versus", "gpu"], as_index=True)["speedup"].apply(gmean).reset_index()
    
    ax = sns.scatterplot(x="size_str", y="speedup", hue="versus", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="versus", hue_order=order, style_order=order, linewidth=0.05)
      
    if sizes is None:
        labels = sorted(data["size"].unique())
    else:
        labels = sizes.copy()
    labels_str = [str(l) for l in labels]
    
    # Set the same y limits in each plot;
    num_y_ticks = 6
    ax.set_ylim((0.5, 3))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                    
    # Set the x ticks;
    xticks = []
    for i, l in enumerate(labels_str):
        xticks += [l]
       
    # ax.set_xticks(xticks)
    ax.set_xticks(range(0, len(xticks), 2))

    ax.set_xticklabels(labels=[get_exp_label(l) for i, l in enumerate(labels) if i % 2 == 0], rotation=0, ha="center", fontsize=9)
    # ax.set_xticklabels(labels=[get_exp_label(l) for i, l in enumerate(labels)], rotation=0, ha="center", fontsize=9)
    ax.tick_params(labelcolor="black", pad=3)

    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(num_y_ticks))
    if y == 0:
        ax.set_yticklabels(labels=["{:.1f}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=10)
    else:
        ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
        # Hide tick markers;
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False) 
            tic.tick2line.set_visible(False) 
            
    # Add policy annotation;
    gpu_dict = {"GTX960": "GTX960", "GTX1660 Super": "GTX1660 Super", "P100": "Tesla P100"}
    if y == 0 and x % 2 == 0:
        ax.annotate(gpu_dict[g], xy=(-0.3, -1.0), fontsize=14, ha="center", xycoords="axes fraction", rotation=90) 
    
    ax.set_ylabel(None)     
    ax.set_xlabel(None) 
    
    # Add benchmark name and baseline execution time annotations;
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.08), fontsize=10, ha="center", xycoords="axes fraction")
    
    # Turn off tick lines;
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    # ax.tick_params(axis="x", which="major",length=3)
    
    # Add baseline execution time annotations (median of execution time across blocks);
    gpu_dict = {"GTX960": "960", "GTX1660 Super": "1660", "P100": "P100"}
    palette_gpu = [COLORS["peach1"], COLORS["b8"], COLORS["b2"]]
    ax.annotate("Median GrCUDA exec. time (ms):", xy=(0, -0.45), fontsize=9, ha="left", xycoords="axes fraction", color="#949494")
    if (j == 0):
        ax.annotate(f"{gpu_dict[g]}:", xy=(-0.75, -0.61), fontsize=9, color="#949494", ha="right", xycoords=("data", "axes fraction"))
    for l_i, l in enumerate(labels):
        vals = data[(data["size"] == int(l))]["computation_sec"]
        baseline_median = np.median(vals) if len(vals) > 0 else np.nan
        if not math.isnan(baseline_median) and l_i % 2 == 0:
            ax.annotate(f"{int(1000 * baseline_median)}", xy=(l_i, -0.61), fontsize=9, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
        
    return ax


def build_exec_time_plot_grcuda_cuda_2rows_multigpu3(data, gridspec, x, y, exec_policy, palette_in, markers_in):
    
    # data = pd.melt(data, id_vars=["benchmark", "size", "block_size_str", "computation_sec"], value_vars=data.columns[-3:],
    #     var_name="versus", value_name="speedup")
    # data["size_str"] = data["size"].astype(str)
    
    if exec_policy == ASYNC_POLICY_NAME:
        data = data[~data["versus"].isin(["speedup_sync", "speedup_cudagraphsingle", "speedup_cudagraph"])]
        if len(data) == 0:
            return
    elif exec_policy == "sync":
        data = data[data["versus"].isin(["speedup_cudagraphsingle"])]
        if len(data) == 0:
            return
    else:
        raise ValueError(exec_policy + " is not ok!")
    # data = data[data["versus"] != "speedup_sync"]
    # print(x,y,exec_policy,len(data))
    legend_labels = ["Serial Scheduler", "Paraller Scheduler"]
    
    order_p = data["versus"].unique()
    order_m = data["gpu"].unique()
    palette = [palette_in[o] for o in order_p]
    markers = [markers_in[o] for o in order_m]
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(gridspec[x, y])
    ax.axhspan(0, 1, facecolor='0.8', alpha=0.3)

    ax = sns.lineplot(x="size_str", y="speedup", hue="versus", data=data[data["gpu"] == "GTX960"], palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, hue_order=order_p, zorder=2)
    ax = sns.lineplot(x="size_str", y="speedup", hue="versus", data=data[data["gpu"] == "GTX1660 Super"], palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, hue_order=order_p, zorder=2)
    ax = sns.lineplot(x="size_str", y="speedup", hue="versus", data=data[data["gpu"] == "P100"], palette=palette, ax=ax, estimator=gmean,
                      err_style="bars", linewidth=2, legend=None, sort=False, ci=None, hue_order=order_p, zorder=2)
    data_averaged = data.groupby(["size_str", "versus", "gpu"], as_index=True)["speedup"].apply(gmean).reset_index()
    
    ax = sns.scatterplot(x="size_str", y="speedup", data=data_averaged[data_averaged["gpu"] == "GTX960"], color="#ffffff", ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers_in, linewidth=0.08, style="gpu")
    ax = sns.scatterplot(x="size_str", y="speedup", data=data_averaged[data_averaged["gpu"] == "GTX1660 Super"], color="#ffffff", ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers_in, linewidth=0.08, style="gpu")
    ax = sns.scatterplot(x="size_str", y="speedup", data=data_averaged[data_averaged["gpu"] == "P100"], color="#ffffff", ax=ax, edgecolor="#0f0f0f",
          size_norm=30, legend=False, zorder=3, ci=None, markers=markers_in, linewidth=0.08, style="gpu")
    
    labels = sorted(data["size"].unique())
    labels_str = [str(l) for l in labels]
    
    # Set the same y limits in each plot;
    num_y_ticks = 6
    if exec_policy == "sync":
        ax.set_ylim((0.5, 1.5))
        num_y_ticks = 5
    elif exec_policy == ASYNC_POLICY_NAME and x == 3:
        ax.set_ylim((0.5, 2.0))
        num_y_ticks = 4
    else:
        ax.set_ylim((0.5, 3))

    # Add a horizontal line to denote speedup = 1x;
    ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
                
    # Add a vertical line to split GPUs;
    max_size_960 = str(data[data["gpu"] == "GTX960"]["size"].max())
    
    # Set the x ticks;
    odd_ticks = 0 if (len(labels_str) % 2 == 1) else 1
    xticks = []
    max_tick_960 = 0
    for i, l in enumerate(labels_str):
        if i % 2 == odd_ticks:
            xticks += [l]
        if l == max_size_960:
            max_tick_960 = i
    # ax.axvline(x=max_tick_960, color="#2f2f2f", linestyle="--", zorder=1, linewidth=0.5, alpha=0.5)
    # ax.annotate("GTX960", xy=(0.35, 0.85), fontsize=8, ha="center", xycoords="axes fraction", color="#2f2f2f", alpha=0.5)
    # ax.annotate("P100", xy=(0.6, 0.85), fontsize=8, ha="center", xycoords="axes fraction", color="#2f2f2f", alpha=0.5)        
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels=[get_exp_label(l) for i, l in enumerate(labels) if i % 2 == odd_ticks], rotation=0, ha="center", fontsize=9)
    ax.tick_params(labelcolor="black", pad=3)

    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(num_y_ticks))
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
    ax.annotate(f"{BENCHMARK_NAMES[data['benchmark'].iloc[0]]}", xy=(0.50, 1.08), fontsize=10, ha="center", xycoords="axes fraction")
    
    # Turn off tick lines;
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    # ax.tick_params(axis="x", which="major",length=3)
    
    # Add baseline execution time annotations (median of execution time across blocks);
    gpus = ["960", "1660", "P100"]
    palette_gpu = [COLORS["peach1"], COLORS["b8"], COLORS["b2"]]
    ax.annotate("Median GrCUDA exec. time (ms):", xy=(0, -0.42), fontsize=9, ha="left", xycoords="axes fraction", color="#949494")
    for g_i, gpu in enumerate(data["gpu"].unique()):
        if g_i < len(gpus):
            if (j == 0):
                ax.annotate(f"{gpus[g_i]}:", xy=(-0.75, -0.57 - g_i * 0.15), fontsize=9, color=palette_gpu[g_i], ha="right", xycoords=("data", "axes fraction"))

            for l_i, l in enumerate(labels):
                vals = data[(data["size"] == int(l)) & (data["gpu"] == gpu)]["computation_sec"]
                baseline_median = np.median(vals) if len(vals) > 0 else np.nan
                # print(i, j, gpu, baseline_median)
                if not math.isnan(baseline_median) and l_i % 2 == odd_ticks:
                    ax.annotate(f"{int(1000 * baseline_median)}", xy=(l_i, -0.37 - g_i * 0.1), fontsize=9, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
 
    # # Add baseline execution time annotations (median of execution time across blocks);
    # ax.annotate("Median GrCUDA exec. time (ms):", xy=(0, -0.42), fontsize=8, ha="left", xycoords="axes fraction", color=COLORS["peach1"])
    # for i, l in enumerate(labels):    
    #     baseline_median = np.median(data[data["size"] == int(l)]["computation_sec"])
    #     ax.annotate(f"{int(1000 * baseline_median)}", xy=(i, -0.57 - j * 0.1), fontsize=8, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
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
    data = data[(data["block_size_1d"] == 256) & (data["exec_policy"] == ASYNC_POLICY_NAME)].copy()
    # data = data[(data["exec_policy"] ==ASYNC_POLICY_NAME)].copy()

    # For each benchmark, keep the data relative to the largest data size;
    biggest_sizes = data.groupby(["benchmark"])["size"].max().to_dict()
    data_filtered = []
    for k, v in biggest_sizes.items():
        data_filtered += [data[(data["benchmark"] == k) & (data["size"] == v)]]
    data = pd.concat(data_filtered).reset_index(drop=True)
    
    # Normalize execution times so that the CUDA baseline has median 1;
    data["normalized_time_cuda"] = 1
    data["normalized_time_grcuda"] = 1
    
    # grouped_data = data.groupby(["benchmark", "size", "block_size_str"], as_index=False)
    grouped_data = data.groupby(["benchmark"], as_index=False, sort=False)
    for group_key, group in grouped_data:
        # Compute the median baseline computation time;
        median_baseline = np.median(group["computation_sec_cuda"])
        # Compute the speedup for this group;
        data.loc[group.index, "normalized_time_cuda"] = group["computation_sec_cuda"].values / median_baseline
        data.loc[group.index, "normalized_time_grcuda"] = group["computation_sec_grcuda"].values / median_baseline
        
    benchmarks = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    block_sizes = data["block_size_str"].unique()
    sizes = data["size"].unique()
            
    # Initialize the plot;
    g = sns.FacetGrid(data, row="benchmark", aspect=5, height=1.2, sharey=False, sharex=False,)

    # Plot a vertical line corresponding to speedup = 1;
    g.map(plt.axvline, x=1, lw=0.75, clip_on=True, zorder=0, linestyle="--", ymax=0.5)         
    # Plot the densities. Plot them twice as the second time we plot just the white contour;     
    g.map(sns.kdeplot, "normalized_time_cuda", clip_on=False, shade=True, alpha=0.6, lw=1, color=COLORS["peach1"], zorder=2)  
    g.map(sns.kdeplot, "normalized_time_grcuda", clip_on=False, shade=True, alpha=0.6, lw=1, color=COLORS["b8"], zorder=2)
    g.map(sns.kdeplot, "normalized_time_cuda", clip_on=False, color="w", lw=1.1, zorder=2)
    g.map(sns.kdeplot, "normalized_time_grcuda", clip_on=False, color="w", lw=1.1, zorder=2)
    # Plot the horizontal line below the densities;
    g.map(plt.axhline, y=0, lw=0.75, clip_on=False, zorder=5, color="0.6")
    
    # Write the x-axis tick labels using percentages;
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{x:.2f}x"
    # Fix the horizontal axes.
    # For each benchmark, find the smallest and largest values;    
    offsets = {
        "b1": [0.95, 1.02],
        "b5": [0.95, 1.05],
        "b6": [0.95, 1.05],
        "b7": [0.99, 0.99],
        "b8": [0.87, 1.13],
        "b10": [0.87, 1.13],
        }
    # offsets = {
    #     "b1": [0.85, 1.15],
    #     "b5": [0.95, 1.05],
    #     "b6": [0.95, 1.05],
    #     "b7": [0.98, 1.02],
    #     "b8": [0.87, 1.13]}
    
    for i, ax in enumerate(g.axes[:, 0]):
        b = benchmarks[i]
        d = data[data["benchmark"] == b]
        max_v = offsets[b][1] * max(d["normalized_time_grcuda"].max(), d["normalized_time_cuda"].max())
        min_v = offsets[b][0] * min(d["normalized_time_grcuda"].min(), d["normalized_time_cuda"].min()) 
        print(min_v, max_v)
        ax.set_xlim(left=min_v, right=max_v)
        ax.xaxis.set_major_formatter(major_formatter)
    
    # Titles and labels;
    g.set_titles("")
    g.set(xlabel=None)
    
    # Add block size labels;
    for i, ax in enumerate(g.axes[-1]):
        ax.annotate("1D={}, 2D={}".format(*block_sizes[i].split(",")), xy=(0.5, -0.8), xycoords="axes fraction", ha="center", color="#2f2f2f", fontsize=14)    
    for i, ax in enumerate(g.axes[:, 0]):
        # ax.annotate(f"{get_exp_label(sizes[i])}", xy=(-0.1, 0.05), xycoords="axes fraction", ha="center", color="#2f2f2f", fontsize=14)      
        ax.annotate(f"{BENCHMARK_NAMES[benchmarks[i]]}", xy=(0.0, 0.09), xycoords="axes fraction", ha="left", color="#2f2f2f", fontsize=12)    

    # Fix the borders. This must be done here as the previous operations update the default values;
    g.fig.subplots_adjust(top=0.83,
                      bottom=0.15,
                      right=0.95,
                      left=0.05,
                      hspace=0.4,
                      wspace=0.1)
    
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    # Add custom legend;
    custom_lines = [Patch(facecolor=COLORS["peach1"], edgecolor="#2f2f2f", label="CUDA"),
                    Patch(facecolor=COLORS["b8"], edgecolor="#2f2f2f", label="GrCUDA"),
                    ]
    leg = g.fig.legend(custom_lines, ["CUDA", "GrCUDA"], bbox_to_anchor=(0.97, 0.98), fontsize=12)
    leg.set_title(None)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    # Main plot title;
    g.fig.suptitle("Exec. Time Distribution,\nCUDA vs GrCUDA", ha="left", x=0.05, y=0.95, fontsize=18)
    
    return g    

#%%

##############################
##############################

if __name__ == "__main__":
    # data_grcuda = load_data(INPUT_DATE_GRCUDA, skip_iter=3)
    # data_cuda = load_data_cuda(INPUT_DATE_CUDA, skip_iter=3)
    # data = join_tables(data_grcuda, data_cuda)
    
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
    # plt.subplots_adjust(top=0.8,
    #                 bottom=0.15,
    #                 left=0.2,
    #                 right=0.90,
    #                 hspace=1.1,
    #                 wspace=0.15)
        
    # exec_time_axes = []
    # for b_i, b in enumerate(benchmark_list):
    #     for block_size_i, block_size in enumerate(block_size_list): 
    #         curr_res = data[(data["benchmark"] == b) & (data["block_size_str"] == block_size)].reset_index(drop=True)  
    #         exec_time_axes += [build_exec_time_plot_grcuda_cuda(curr_res, gs, block_size_i, b_i)]
            
    # plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=20, ha="center", va="center", xycoords="figure fraction")
    # plt.annotate("Speedup", xy=(0.02, 0.5), fontsize=20, ha="center", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Speedup of GrCUDA w.r.t. CUDA", fontsize=25, x=.05, y=0.99, ha="left")
    
    # save_plot(PLOT_DIR, "speedup_baseline_grcuda_cuda_{}.{}", OUTPUT_DATE)

    
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
    #     for p_i, p in enumerate(policy_list): 
    #         curr_res = data[(data["benchmark"] == b) & (data["exec_policy"] == p)].reset_index(drop=True)  
    #         exec_time_axes += [build_exec_time_plot_grcuda_cuda_compact(curr_res, gs, p_i, b_i)]
        
    # plt.annotate("Input number of elements", xy=(0.5, 0.03), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # plt.annotate("Speedup", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Speedup of GrCUDA w.r.t. CUDA", fontsize=25, x=.05, y=0.99, ha="left")
    
    # save_plot(PLOT_DIR, "speedup_baseline_grcuda_cuda_compact_{}.{}", OUTPUT_DATE)
    
    
    # %% Similar plot, but the plot fits on 1 row of a paper;
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
    # fig = plt.figure(figsize=(2.2 * num_col, 1.8 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.84,
    #                 bottom=0.12,
    #                 left=0.10,
    #                 right=0.98,
    #                 hspace=1.1,
    #                 wspace=0.15)
        
    # exec_time_axes = []
    # for p_i, p in enumerate(policy_list): 
    #     for b_i, b in enumerate(benchmark_list):
    #         index_tot = (len(benchmark_list) * p_i + b_i)
    #         j = index_tot % num_col
    #         i = index_tot // num_col
    #         curr_res = data[(data["benchmark"] == b) & (data["exec_policy"] == p)].reset_index(drop=True)  
    #         curr_res = remove_outliers_df_grouped(curr_res, column="grcuda_cuda_speedup", group=["block_size_str", "size"])
    #         exec_time_axes += [build_exec_time_plot_grcuda_cuda_2rows(curr_res, gs, i, j)]
        
    # plt.annotate("Input number of elements", xy=(0.5, 0.02), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # # plt.annotate("Speedup", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Speedup of GrCUDA scheduling w.r.t.\nhand-optimized C++ CUDA scheduling", fontsize=16, x=.05, y=0.99, ha="left")
    
    # l1 = lines.Line2D([0.01, 0.99], [0.465, 0.465], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--", linewidth=1)
    # fig.lines.extend([l1])
    
    # save_plot(PLOT_DIR, "speedup_baseline_grcuda_cuda_2rows_{}.{}", OUTPUT_DATE)
    
    #%% Similar plot, but using multiple CUDA benchmarks types;
    ###############################
    ###############################
     
    # data_grcuda_p100 = load_data(INPUT_DATE_GRCUDA_P100, skip_iter=3)
    # data_grcuda_960 = load_data(INPUT_DATE_GRCUDA_960, skip_iter=3)
    # data_cuda_960 = load_data_cuda(INPUT_DATE_CUDA_960, skip_iter=3, add_prefetch_as_policy=False)
    # data_cuda_p100 = load_data_cuda(INPUT_DATE_CUDA_P100, skip_iter=3, add_prefetch_as_policy=False)
    # data_cuda_960["gpu"] = "GTX960"
    # data_grcuda_960["gpu"] = "GTX960"
    # data_cuda_p100["gpu"] = "P100"
    # data_grcuda_p100["gpu"] = "P100"
    
    # data_grcuda_p100 = data_grcuda_p100[data_grcuda_p100["force_prefetch"] == False]
    # data_grcuda_960 = data_grcuda_960[data_grcuda_960["force_prefetch"] == False]
    # data_cuda_960 = data_cuda_960[data_cuda_960["force_prefetch"] == False]
    # data_cuda_p100 = data_cuda_p100[data_cuda_p100["force_prefetch"] == False]
    
    # # Ignore sync policies;
    # # data_cuda_960 = data_cuda_960[data_cuda_960["exec_policy"] != "sync"]
    # # data_grcuda_960 = data_grcuda_960[data_grcuda_960["exec_policy"] != "sync"]
    # # data_cuda_p100 = data_cuda_p100[data_cuda_p100["exec_policy"] != "sync"]
    # # data_grcuda_p100 = data_grcuda_p100[data_grcuda_p100["exec_policy"] != "sync"]
    
    # data_960 = join_tables_baseline(data_cuda_960, data_grcuda_960)
    # data_p100 = join_tables_baseline(data_cuda_p100, data_grcuda_p100)
    
    # data = pd.concat([data_960, data_p100]).reset_index(drop=True)
    
    # # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    # sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # plt.rcParams['axes.titlepad'] = 20 
    # plt.rcParams['axes.labelpad'] = 10 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    # plt.rcParams['xtick.major.pad'] = 4
    
    # data = pd.melt(data, id_vars=["gpu", "benchmark", "exec_policy", "size", "block_size_str", "computation_sec"], value_vars=data.columns[-5:],
    #     var_name="versus", value_name="speedup")
    # data["size_str"] = data["size"].astype(str)
    
    # palette = {f"speedup_{ASYNC_POLICY_NAME}": COLORS["peach1"], "speedup_cudagraph": COLORS["b2"], "speedup_sync":  COLORS["b8"], "speedup_cudagraphmanual":  COLORS["b4"],  "speedup_cudagraphsingle":  COLORS["b8"]}
    # markers = {f"speedup_{ASYNC_POLICY_NAME}": "o", "speedup_cudagraph": "X", "speedup_sync": "D", "speedup_cudagraphmanual": "P", "speedup_cudagraphsingle": "D"}
    
    # #%%
    
    # # Lists of benchmarks and block sizes;
    # benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    # policy_list = list(reversed(sorted(data["exec_policy"].unique())))
    # num_col = len(benchmark_list) // 2
    # num_row = len(policy_list) * 2
    # fig = plt.figure(figsize=(2.2 * num_col, 2.15 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.84,
    #                 bottom=0.12,
    #                 left=0.10,
    #                 right=0.98,
    #                 hspace=1.1,
    #                 wspace=0.15)
    
    # # Keep only 1 versus;
    # # data = data[data["versus"] == "speedup_cudagraph"]
           
    # exec_time_axes = []
    # for p_i, p in enumerate(policy_list): 
    #     for b_i, b in enumerate(benchmark_list):
    #         index_tot = (len(benchmark_list) * p_i + b_i)
    #         j = index_tot % num_col
    #         i = index_tot // num_col
    #         curr_res = data[(data["benchmark"] == b) & (data["exec_policy"] == p)].reset_index(drop=True)  
    #         curr_res = remove_outliers_df_grouped(curr_res, column="computation_sec", group=["block_size_str", "size"])
    #         exec_time_axes += [build_exec_time_plot_grcuda_cuda_2rows_multigpu(curr_res, gs, i, j, p, palette, markers)]
        
    # # Legend; 
    # versus = [l for l in data["versus"].unique() if l != "speedup_sync"]
    # names = {f"speedup_{ASYNC_POLICY_NAME}": "Hand-tuned CUDA events", "speedup_cudagraph": "CUDA Graphs + events", "speedup_sync": "CUDA synchronous", "speedup_cudagraphmanual": "CUDA Graphs, manual dep.", "speedup_cudagraphsingle": "CUDA Graphs, single stream"}
    # legend_labels = [names[l] for l in versus]
    # custom_lines = [
    #     lines.Line2D([], [], color="white", marker=markers[l], markersize=10, label=names[l], markerfacecolor=palette[l], markeredgecolor="#2f2f2f") 
    #     for l in versus]        
    # leg = fig.legend(custom_lines, legend_labels, 
    #                           bbox_to_anchor=(0.99, 1), fontsize=10, ncol=1, handletextpad=0.1, columnspacing=0.2)
    # leg.set_title("CUDA baseline type", prop={"size": 10})
    # leg._legend_box.align = "left"    
        
    # plt.annotate("Input number of elements (x-axis not to scale)", xy=(0.5, 0.02), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # # plt.annotate("Speedup", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Speedup of our GrCUDA scheduling\nagainst hand-optimized CUDA Graphs\n(higher is better)", fontsize=16, x=.05, y=0.99, ha="left")
    
    # l1 = lines.Line2D([0.01, 0.99], [0.455, 0.455], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--", linewidth=1)
    # fig.lines.extend([l1])
    
    # save_plot(PLOT_DIR, "speedup_baseline_grcuda_cuda_multicuda_{}.{}", OUTPUT_DATE)
    
    
    #%% Ridge plot with distributions;
    # g = ridgeplot(data)
    # save_plot(PLOT_DIR, "speedup_baseline_grcuda_cuda_ridgeplot_{}.{}", OUTPUT_DATE)
    
    
    # %% Summary plot with CUDA speedups;
    ###################################
    ###################################
   
    # BENCHMARK_NAMES = {"b1": "VEC", "b5": "B&S", "b8": "Images", "b6": "ML", "b7": "HITS", "b10": "DL", "mean": "", "mean2": ""}

    # sns.set_style("white", {"ytick.left": True})
    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # plt.rcParams['axes.titlepad'] = 25 
    # plt.rcParams['axes.labelpad'] = 5 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    # plt.rcParams['xtick.major.pad'] = 2
    
    # data_cuda_960 = load_data_cuda(INPUT_DATE_CUDA_960, skip_iter=3)
    # data_cuda_p100 = load_data_cuda(INPUT_DATE_CUDA_P100, skip_iter=3)
    # data_cuda_1660 = load_data_cuda(INPUT_DATE_CUDA_1660, skip_iter=3)
    # gpus = ["GTX960", "GTX1660 Super", "P100"][1:]
    # # data_cuda_960["gpu"] = gpus[0]
    # data_cuda_1660["gpu"] = gpus[0]
    # data_cuda_p100["gpu"] = gpus[1]
        
    # data_list = []
    # gmean_horizontal_values = []
    # for data_c in [data_cuda_1660, data_cuda_p100]:
    #     data_cuda_2 = remove_outliers_df_grouped(data_c, column="computation_speedup", group=["benchmark", "exec_policy_full", "block_size_str", "size", "gpu"])
    #     cuda_summary = data_cuda_2[data_cuda_2["exec_policy_full"] == ASYNC_POLICY_NAME].groupby(["benchmark", "block_size_str", "size", "gpu"], sort=False)["computation_speedup"].apply(gmean).reset_index(drop=False)
    #     cuda_summary = cuda_summary.sort_values(by=["benchmark"], key=lambda x: x.apply(lambda y: int(y[1:])))              
        
    #     # Add geomean;
    #     gmean_res = pd.DataFrame(cuda_summary.groupby(["benchmark"], as_index=False).agg(gmean))
    #     gmean_res["benchmark"] = "mean"
    #     gmean_horizontal_value = gmean(gmean_res["computation_speedup"])
    #     gmean_horizontal_values += [gmean_horizontal_value]
    #     gmean_res["computation_speedup"] = 0
    #     res_tmp = pd.concat([cuda_summary, gmean_res])
        
    #     # Do it again, workaround to have another fake column;
    #     gmean_res = pd.DataFrame(cuda_summary.groupby(["benchmark"], as_index=False).agg(gmean))
    #     gmean_res["benchmark"] = "mean2"
    #     gmean_horizontal_value = gmean(gmean_res["computation_speedup"])
    #     gmean_res["computation_speedup"] = 0
    #     data_list += [res_tmp, gmean_res]
    # res = pd.concat(data_list).reset_index(drop=True)
    
    # num_col = 1
    # fig = plt.figure(figsize=(3.8 * num_col, 2))
    # gs = gridspec.GridSpec(1, 1)
    # plt.subplots_adjust(top=0.78,
    #                 bottom=0.15,
    #                 left=0.14,
    #                 right=.99,
    #                 hspace=0.9,
    #                 wspace=0.05)
    
    # palettes = ["#A2F2B1", "#6CC982"]# * len(cuda_summary["benchmark"].unique()) + ["#96DE9B"]
  
    # ax = fig.add_subplot(gs[0, 0])
    # ax0 = ax
    
    # ax = sns.barplot(x="benchmark", y="computation_speedup", hue="gpu", data=res, order=list(BENCHMARK_NAMES.keys()), ci=95,
    #                   palette=palettes, capsize=.05, errwidth=0.8, ax=ax, edgecolor="#2f2f2f", estimator=gmean, zorder=2, saturation=1)
    # ax.legend_.remove()  # Hack to remove legend;
    
    # gpu_dict = {"P100": "P100", "GTX1660 Super": "1660"}
    # for i, g in enumerate(gpus):
    #     ax.axhline(y=float(f"{gmean_horizontal_values[i]:.2}"), color="#D98159" if i else COLORS["peach1"], linestyle="-", zorder=1, linewidth=1, )
    #     color = "#D98159" if i else COLORS["peach1"]
    #     alpha = 1
    #     color = "#2f2f2f"
    #     alpha = 0.75 + i * 0.25
    #     ax.annotate(f"{gpu_dict[g]}, geomean\nspeedup: {gmean_horizontal_values[i]:.2f}x", xy=(0.75, 0.26 + i * 0.25), xycoords="axes fraction", ha="left", alpha=alpha, color=color, fontsize=6)   
    # ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
    # ax.annotate(f"Serial execution", xy=(0.75, 0.12), xycoords="axes fraction", ha="left", color="#2f2f2f", fontsize=6, alpha=0.5)   
    
    # ax.set_ylabel("Speedup", fontsize=11)
    # ax.set_xlabel("")
    # ax.set_ylim((0.5, 3))
    # labels = ax.get_xticklabels()
    # for j, l in enumerate(labels):
    #     l.set_text(BENCHMARK_NAMES[l._text])
    # ax.set_xticklabels(labels, ha="center", va="top")
    # ax.tick_params(axis='x', which='major', labelsize=8, rotation=0)
    
    # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))
    # ax.yaxis.set_major_locator(plt.LinearLocator(6))
    # ax.tick_params(axis='y', which='major', labelsize=8)
    # ax.grid(True, axis="y")
    
    # update_width(ax, 0.4)
    
    # # Speedup labels;
    # offsets = []
    # for k, g in res.groupby(["benchmark", "gpu"]):
    #     offsets += [get_upper_ci_size(g["computation_speedup"], ci=0.5)]
    # offsets = offsets[:(len(offsets)//2)] + ([0] * 2) + offsets[(len(offsets)//2):] + ([0] * 2)
    # offsets = [o + 0.05 if not np.isnan(o) else 0.2 for o in offsets]
    # offsets[0] = 0.15
    # # offsets[5] = 0.1
    # # offsets[6] = 0.1
    # # offsets[8] = 0.2
    # offsets[9] = 0.12
    # offsets[10] = 0.1
    # add_labels(ax, vertical_offsets=offsets, rotation=0, format_str="{:.2f}", fontsize=6, skip_zero=False)
    
    # plt.suptitle("Achievable speedup in C++ CUDA with hand-tuned\nGPU data transfer and execution overlap", fontsize=11, x=.01, y=0.99, ha="left")
    
    # gpu_dict = {"P100": "Tesla P100", "GTX1660 Super": "GTX1660 Super"}
    # legend_labels = [gpu_dict[g] for g in gpus]
    # custom_lines = [Patch(facecolor=palettes[i], edgecolor="#2f2f2f", label=l)
    #                 for i, l in enumerate(legend_labels)]
    # leg = fig.legend(custom_lines, legend_labels, bbox_to_anchor=(0.99, 0.78), fontsize=8, ncol=1)
    # leg.set_title("")
    # leg._legend_box.align = "left"
    # leg.get_frame().set_facecolor('white')
        
    # save_plot(PLOT_DIR, "cuda_speedup_{}.{}", OUTPUT_DATE)
    
    #%% Using 3 GPUs
    ############################
    ############################
        
    # data_grcuda_p100 = load_data(INPUT_DATE_GRCUDA_P100, skip_iter=3)
    # data_grcuda_960 = load_data(INPUT_DATE_GRCUDA_960, skip_iter=3)
    # data_grcuda_1660 = load_data(INPUT_DATE_GRCUDA_1660, skip_iter=3)
    # data_cuda_960 = load_data_cuda(INPUT_DATE_CUDA_960, skip_iter=3)
    # data_cuda_p100 = load_data_cuda(INPUT_DATE_CUDA_P100, skip_iter=3)
    # data_cuda_1660 = load_data_cuda(INPUT_DATE_CUDA_1660, skip_iter=3)
    # data_cuda_960["gpu"] = "GTX960"
    # data_grcuda_960["gpu"] = "GTX960"
    # data_cuda_p100["gpu"] = "P100"
    # data_grcuda_p100["gpu"] = "P100"
    # data_cuda_1660["gpu"] = "GTX1660 Super"
    # data_grcuda_1660["gpu"] = "GTX1660 Super"
    
    # # Ignore sync policies;
    # # data_cuda_960 = data_cuda_960[data_cuda_960["exec_policy"] != "sync"]
    # # data_grcuda_960 = data_grcuda_960[data_grcuda_960["exec_policy"] != "sync"]
    # # data_cuda_p100 = data_cuda_p100[data_cuda_p100["exec_policy"] != "sync"]
    # # data_grcuda_p100 = data_grcuda_p100[data_grcuda_p100["exec_policy"] != "sync"]
    
    # data_960 = join_tables_baseline(data_cuda_960, data_grcuda_960)
    # data_p100 = join_tables_baseline(data_cuda_p100, data_grcuda_p100)
    # data_1660 = join_tables_baseline(data_cuda_1660, data_grcuda_1660)
    
    # data = pd.concat([data_960, data_1660, data_p100]).reset_index(drop=True)
    
    # data = data[data["force_prefetch"] == False]
    
    # # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    # sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # plt.rcParams['axes.titlepad'] = 20 
    # plt.rcParams['axes.labelpad'] = 10 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    # plt.rcParams['xtick.major.pad'] = 4
    
    # data = pd.melt(data, id_vars=["gpu", "benchmark", "exec_policy", "size", "block_size_str", "computation_sec"], value_vars=data.columns[-5:],
    #     var_name="versus", value_name="speedup")
    # data["size_str"] = data["size"].astype(str)
    
    # palette = {f"speedup_{ASYNC_POLICY_NAME}": COLORS["peach1"], "speedup_cudagraph": COLORS["b2"], "speedup_sync":  COLORS["b8"], "speedup_cudagraphmanual":  COLORS["b4"],  "speedup_cudagraphsingle":  COLORS["b8"]}
    # # markers = {f"speedup_{ASYNC_POLICY_NAME}": "o", "speedup_cudagraph": "X", "speedup_sync": "D", "speedup_cudagraphmanual": "P", "speedup_cudagraphsingle": "D"}
    # markers = {"GTX960": "o", "GTX1660 Super": "X", "P100": "D"}

    # # Lists of benchmarks and block sizes;
    # benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    # policy_list = list(reversed(sorted(data["exec_policy"].unique())))
    # num_col = len(benchmark_list) // 2
    # num_row = len(policy_list) * 2
    # fig = plt.figure(figsize=(2.2 * num_col, 2.15 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.84,
    #                 bottom=0.12,
    #                 left=0.10,
    #                 right=0.98,
    #                 hspace=1.1,
    #                 wspace=0.15)
    
    # # Keep only 1 versus;
    # # data = data[data["versus"] == "speedup_cudagraph"]
           
    # exec_time_axes = []
    # for p_i, p in enumerate(policy_list): 
    #     for b_i, b in enumerate(benchmark_list):
    #         index_tot = (len(benchmark_list) * p_i + b_i)
    #         j = index_tot % num_col
    #         i = index_tot // num_col
    #         curr_res = data[(data["benchmark"] == b) & (data["exec_policy"] == p)].reset_index(drop=True)  
    #         curr_res = remove_outliers_df_grouped(curr_res, column="computation_sec", group=["block_size_str", "size"])
    #         exec_time_axes += [build_exec_time_plot_grcuda_cuda_2rows_multigpu3(curr_res, gs, i, j, p, palette, markers)]
        
    # # Legend; 
    # # versus = [l for l in data["versus"].unique() if l not in ["speedup_sync", "speedup_cudagraph"]]
    # # names = {f"speedup_{ASYNC_POLICY_NAME}": "Hand-tuned CUDA events", "speedup_cudagraph": "CUDA Graphs + events", "speedup_sync": "CUDA synchronous", "speedup_cudagraphmanual": "CUDA Graphs, manual dep.", "speedup_cudagraphsingle": "CUDA Graphs, single stream"}
    # # legend_labels = [names[l] for l in versus]
    # # custom_lines = [
    # #     lines.Line2D([], [], color="white", marker=markers[l], markersize=10, label=names[l], markerfacecolor=palette[l], markeredgecolor="#2f2f2f") 
    # #     for l in versus]        
    # # leg = fig.legend(custom_lines, legend_labels, 
    # #                           bbox_to_anchor=(0.99, 1), fontsize=10, ncol=1, handletextpad=0.1, columnspacing=0.2)
    # # leg.set_title("CUDA baseline type", prop={"size": 10})
    # # leg._legend_box.align = "left"    
        
    # plt.annotate("Input number of elements", xy=(0.5, 0.02), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # # plt.annotate("Speedup", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    # plt.suptitle("Speedup of GrCUDA against\nhand-optimized CUDA Graphs\n(higher is better)", fontsize=16, x=.05, y=0.99, ha="left")
    
    # l1 = lines.Line2D([0.01, 0.99], [0.455, 0.455], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--", linewidth=1)
    # fig.lines.extend([l1])
    
    # save_plot(PLOT_DIR, "speedup_baseline_grcuda_cuda_multicuda3_{}.{}", OUTPUT_DATE)
    
    
    
    #%% Performance of GrCUDA vs CUDA Graphs on all GPUs
    ####################################################
    ####################################################    
    
    data_grcuda_p100 = load_data(INPUT_DATE_GRCUDA_P100, skip_iter=3)
    data_grcuda_1660 = load_data(INPUT_DATE_GRCUDA_1660, skip_iter=3)
    data_grcuda_960 = load_data(INPUT_DATE_GRCUDA_960, skip_iter=3)
    data_cuda_960 = load_data_cuda(INPUT_DATE_CUDA_960, skip_iter=3, add_prefetch_as_policy=False)
    data_cuda_1660 = load_data_cuda(INPUT_DATE_CUDA_1660, skip_iter=3, add_prefetch_as_policy=False)
    data_cuda_p100 = load_data_cuda(INPUT_DATE_CUDA_P100, skip_iter=3, add_prefetch_as_policy=False)
    data_cuda_960["gpu"] = "GTX960"
    data_grcuda_960["gpu"] = "GTX960"
    data_cuda_1660["gpu"] = "GTX1660 Super"
    data_grcuda_1660["gpu"] = "GTX1660 Super"
    data_cuda_p100["gpu"] = "P100"
    data_grcuda_p100["gpu"] = "P100"
    
    data_grcuda_p100 = data_grcuda_p100[data_grcuda_p100["force_prefetch"] == False]
    data_grcuda_960 = data_grcuda_960[data_grcuda_960["force_prefetch"] == False]
    data_grcuda_1660 = data_grcuda_1660[data_grcuda_1660["force_prefetch"] == False]
    data_cuda_960 = data_cuda_960[data_cuda_960["force_prefetch"] == False]
    data_cuda_p100 = data_cuda_p100[data_cuda_p100["force_prefetch"] == False]
    data_cuda_1660 = data_cuda_1660[data_cuda_1660["force_prefetch"] == False]

    # Ignore sync policies;
    # data_cuda_960 = data_cuda_960[data_cuda_960["exec_policy"] != "sync"]
    # data_grcuda_960 = data_grcuda_960[data_grcuda_960["exec_policy"] != "sync"]
    # data_cuda_p100 = data_cuda_p100[data_cuda_p100["exec_policy"] != "sync"]
    # data_grcuda_p100 = data_grcuda_p100[data_grcuda_p100["exec_policy"] != "sync"]
    
    data_960 = join_tables_baseline(data_cuda_960, data_grcuda_960)
    data_1660 = join_tables_baseline(data_cuda_1660, data_grcuda_1660)
    data_p100 = join_tables_baseline(data_cuda_p100, data_grcuda_p100)
    
    data = pd.concat([data_960, data_1660, data_p100]).reset_index(drop=True)
    
    # sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    plt.rcParams['xtick.major.pad'] = 4
    
    data = pd.melt(data, id_vars=["gpu", "benchmark", "exec_policy", "size", "block_size_str", "computation_sec"], value_vars=data.columns[-5:],
        var_name="versus", value_name="speedup")
    data = data[~data["versus"].isin(["speedup_sync", "speedup_cudagraphsingle"])]
    data["size_str"] = data["size"].astype(str)
    
    data = data[data["exec_policy"] == ASYNC_POLICY_NAME]
    
    palette = {f"speedup_{ASYNC_POLICY_NAME}": COLORS["peach1"], "speedup_cudagraph": COLORS["b2"], "speedup_sync":  COLORS["b8"], "speedup_cudagraphmanual":  COLORS["b4"],  "speedup_cudagraphsingle":  COLORS["b8"]}
    markers = {f"speedup_{ASYNC_POLICY_NAME}": "o", "speedup_cudagraph": "X", "speedup_sync": "D", "speedup_cudagraphmanual": "P", "speedup_cudagraphsingle": "D"}
    
    #%%
    
    # Lists of benchmarks and block sizes;
    benchmark_list = [b for b in BENCHMARK_NAMES.keys() if b in data["benchmark"].unique()]
    policy_list = list(reversed(sorted(data["exec_policy"].unique())))
    gpu_list = list(data["gpu"].unique())
    num_col = len(benchmark_list) // 2
    num_row = len(gpu_list) * 2
    fig = plt.figure(figsize=(2.2 * num_col, 1.6 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.875,
                    bottom=0.085,
                    left=0.10,
                    right=0.965,
                    hspace=1,
                    wspace=0.15)
    
    # Keep only 1 versus;
    # data = data[data["versus"] == "speedup_cudagraph"]
           
    exec_time_axes = []
    for g_i, g in enumerate(gpu_list): 
        for b_i, b in enumerate(benchmark_list):
            index_tot = (len(benchmark_list) * g_i + b_i)
            j = index_tot % num_col
            i = index_tot // num_col
            curr_res = data[(data["benchmark"] == b) & (data["gpu"] == g)].reset_index(drop=True)  
            sizes = sorted(data[data["benchmark"] == b]["size"].unique())
            curr_res = remove_outliers_df_grouped(curr_res, column="computation_sec", group=["block_size_str", "size"])
            print(g, b, len(curr_res))
            exec_time_axes += [build_exec_time_plot_grcuda_cuda_3rows_multigpu(curr_res, gs, i, j, g, palette, markers, sizes)]
        
    # Legend; 
    versus = [l for l in data["versus"].unique() if l != "speedup_sync"]
    names = {f"speedup_{ASYNC_POLICY_NAME}": "Hand-tuned CUDA events", "speedup_cudagraph": "CUDA Graphs + events", "speedup_sync": "CUDA synchronous", "speedup_cudagraphmanual": "CUDA Graphs, manual dep.", "speedup_cudagraphsingle": "CUDA Graphs, single stream"}
    legend_labels = [names[l] for l in versus]
    custom_lines = [
        lines.Line2D([], [], color="white", marker=markers[l], markersize=10, label=names[l], markerfacecolor=palette[l], markeredgecolor="#2f2f2f") 
        for l in versus]        
    leg = fig.legend(custom_lines, legend_labels, 
                              bbox_to_anchor=(0.99, 1), fontsize=10, ncol=1, handletextpad=0.1, columnspacing=0.2)
    leg.set_title("CUDA baseline type", prop={"size": 10})
    leg._legend_box.align = "left"    
        
    plt.annotate("Input number of elements (x-axis not to scale)", xy=(0.5, 0.02), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # plt.annotate("Speedup", xy=(0.022, 0.44), fontsize=14, ha="left", va="center", rotation=90, xycoords="figure fraction")    
    plt.suptitle("Speedup of our GrCUDA scheduling\nagainst hand-optimized CUDA Graphs\n(higher is better)", fontsize=16, x=.05, y=0.99, ha="left")
    
    l1 = lines.Line2D([0.01, 0.99], [0.322, 0.322], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--", linewidth=1)
    l2 = lines.Line2D([0.01, 0.99], [0.608, 0.608], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--", linewidth=1)
    fig.lines.extend([l1, l2])
    
    save_plot(PLOT_DIR, "speedup_baseline_grcuda_cuda_3gpu_{}.{}", OUTPUT_DATE)
    