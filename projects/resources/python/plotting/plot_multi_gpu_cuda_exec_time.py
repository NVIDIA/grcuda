# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 20:51:56 2021

@author: albyr
"""
#%%
import math
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from load_data import PLOT_DIR, load_data_cuda_multigpu
from segretini_matplottini.src.plot_utils import (add_labels, PALETTE_OG, PALETTE_G3, 
                                                  get_exp_label, save_plot, get_upper_ci_size)

##############################
##############################

OUTPUT_DATE = "2022_01_20"

# V100;
V100 = "V100"
V100_RES_FOLDERS = [
    # "2021_10_04_15_13_11_cuda_1gpu_v100",
    # "2021_10_04_15_15_29_cuda_2gpu_v100",
    # "2021_10_04_15_15_49_cuda_4gpu_v100",
    "2022_01_16_18_09_04_cuda_1-2gpu_v100",
    "2022_01_16_18_17_05_cuda_4gpu_v100",
    "2021_10_04_15_33_23_cuda_8gpu_v100",
    ]

# A100;
A100 = "A100"
A100_RES_FOLDERS = [
    "2021_10_18_11_50_56_cuda_1gpu_a100",
    "2021_10_18_12_57_50_cuda_2gpu_a100",
    "2021_10_18_13_21_05_cuda_4gpu_a100",
    "2021_10_18_13_44_18_cuda_8gpu_a100",
    ]
    
##############################
##############################    

def plot_speedup_bars(data_in,
                      gpu,
                      speedup_column="speedup",
                      baseline_is_async: bool=True,
                      keep_only_max_size: bool=False,
                      ylabel: str="Speedup",
                      legend_title: str="Baseline: ASYNC, 1 GPU",
                      legend_baseline_label: str=None,
                      ymax: float=6,
                      yticks: int=7):
    plt.rcdefaults()
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": False})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams['axes.labelpad'] = 5
    plt.rcParams['xtick.major.pad'] = 4.2
    plt.rcParams['ytick.major.pad'] = 1
    plt.rcParams['axes.linewidth'] = 0.5
    
    fig = plt.figure(figsize=(2, 0.95), dpi=600)
    gs = gridspec.GridSpec(1, 1)
    plt.subplots_adjust(top=0.78,
                        bottom=0.17,
                        left=0.12,
                        right=0.99,
                        hspace=0.15,
                        wspace=0.8)
    ax = fig.add_subplot(gs[0, 0])
    
    effective_benchmarks = ["MEAN"] + list(data_in["benchmark"].unique())
    fontsize = 4
    
    # Remove async with 1 GPU, it is the baseline;
    data = data_in[~((data_in["gpus"] == 1) & (data_in["exec_policy"] == ("ASYNC" if baseline_is_async else "SYNC")))]
    
    # Keep only the experiments on the largest dataset;
    if keep_only_max_size:
        max_sizes = data.groupby(["benchmark"])["size"].max().reset_index()
        data = data.merge(max_sizes, how="inner", on=["benchmark", "size"])
        
    # Compute mean of all benchmarks, grouped by number of GPUs;
    data_mean = data.groupby("gpus").mean().reset_index()
    data_mean["benchmark"] = "MEAN"
    new_data = [data_mean]
    new_data += [data]
    data = pd.concat(new_data, ignore_index=True)

    num_gpus = len(data["gpus"].unique())
    palette = PALETTE_OG[:num_gpus]

    ##############
    # Main plot  #
    ##############
    
    ax = sns.barplot(x="benchmark", y=speedup_column, order=effective_benchmarks,
                     hue="gpus",
                     palette=palette,
                     data=data,
                     ci=95, capsize=.05, errwidth=0.3, linewidth=0.3,
                     ax=ax, edgecolor="#2f2f2f", estimator=np.mean, saturation=1, zorder=2)
    ax.legend_.remove()  # Hack to remove legend;
    
    ################
    # Refine style #
    ################

    # Grid and axis limits;
    ax.yaxis.grid(True, linewidth=0.4)
    ax.xaxis.grid(False)
    ax.set_xlim((-0.5, len(data["benchmark"].unique()) - 0.5))

    # Axis limits;
    ax.set_ylim((0, ymax))
    # Color background to represent linear scaling of performance;
    ax.fill_between(ax.get_xlim(), 0, 1, facecolor="0.9", alpha=0.5, zorder=0.4, edgecolor="0.9", linewidth=0.1)
    for i in [1, 2, 4, 8][:num_gpus]:
        ax.axhline(y=i, color="0.6", linestyle="--", zorder=1, linewidth=0.4)

    # Ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(yticks))
    ax.set_yticklabels(labels=[f"{l:.1f}x" for l in ax.get_yticks()], ha="right", fontsize=fontsize)
        
    ax.set_xticks([i for i, l in enumerate(data["benchmark"].unique()) if "empty" not in l])
    ax.tick_params(length=2, width=0.5)
    ax.set_xticklabels(labels=effective_benchmarks, ha="center", va="top", rotation=0, fontsize=fontsize - 0.2)
        
    # Set "MEAN" labels to a different color;
    for i, l in enumerate(ax.get_xticklabels()):
        if "MEAN" in l._text:
            l.set_color(PALETTE_OG[3])
    # Change color of mean patches, start by filtering the ones with height > 0 and then look for labels with "MEAN";
    try:
        patches_to_color = [a for a in ax.patches if a.get_height() > 0]
        for i, p in enumerate(patches_to_color):
            if i % len(effective_benchmarks) == 0:
                p.set_facecolor(sns.desaturate(p._facecolor, 0.6))
    except IndexError as e:
        print(e)
    # Separate mean with a vertical line;
    ax.axvline(x=0.5, color="0.6", linestyle="--", zorder=1, linewidth=0.3)
    
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=1)
    plt.xlabel(None)
    
    # Add speedup labels over bars;
    offsets = []
    for j, g_tmp in data.groupby(["benchmark", "gpus"]):
        offsets += [get_upper_ci_size(g_tmp[speedup_column], ci=0.95)]
    offsets = [o if not np.isnan(o) else 0.05 for o in offsets]
    add_labels(ax, vertical_offsets=offsets, rotation=0, format_str="{:.2f}", fontsize=2.2, skip_zero=False)
    
    # Add label with GPU name;
    ax.annotate(gpu, xy=(0.99, 0.9), xycoords="axes fraction", ha="right", color="#2f2f2f", fontsize=fontsize, alpha=1)   
    
    # Create hierarchical x ticks;
    y_min = -0.08
    y_max = -0.13
    group_labels = effective_benchmarks
    bar_width = ax.patches[0].get_width()  # Get width of a bar
    labels = ["S" if baseline_is_async else "A", 2, 4, 8][:num_gpus]
    for i in range(len(group_labels)):
        x_start = i - bar_width * (num_gpus / 2)
        x_end = i + bar_width * (num_gpus / 2)
        x_middle = (x_start + x_end) / 2
        ax.hlines(y_min, x_start, x_end, color="#2f2f2f", linewidth=0.3, clip_on=False, transform=blended_transform_factory(ax.transData, ax.transAxes))
        ax.vlines(x_middle, y_min, y_max, color="#2f2f2f", linewidth=0.3, clip_on=False, transform=blended_transform_factory(ax.transData, ax.transAxes))
        for l_i, l in enumerate(labels):
            start = bar_width * (num_gpus / 2)
            ax.annotate(l, xy=(i - start + bar_width / 2 + bar_width * l_i, y_min + 0.02),
                        xycoords=blended_transform_factory(ax.transData, ax.transAxes), clip_on=False,
                        fontsize=3.2, ha="center")
 
    # Add legend;
    if legend_baseline_label is None:
        legend_baseline_label = f"{'' if baseline_is_async else 'A'}SYNC, 1 GPU"
    labels = [legend_baseline_label, "2 GPU", "4 GPU", "8 GPU"][:num_gpus]
    patches = [Patch(facecolor=palette[i], edgecolor="#2f2f2f", label=l, linewidth=0.5) for i, l in enumerate(labels)]
    leg = fig.legend(patches, labels, bbox_to_anchor=(0.55, 1.01), fontsize=fontsize,
                     ncol=num_gpus, loc="upper center", handlelength=1.2, 
                     handletextpad=0.2, columnspacing=0.5, title=legend_title, title_fontsize=fontsize)
    leg._legend_box.align = "left"
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_facecolor('white')
    
    return ax


def plot_speedup_line(data_in, gpu,
                      speedup_column: str="speedup",
                      baseline_time_column: str="baseline_time",
                      baseline_is_async: bool=True,
                      kind: str="CUDA"):
    
    # Remove async with 1 GPU, it is the baseline;
    data = data_in[~((data_in["gpus"] == 1) & (data_in["exec_policy"] == ("ASYNC" if baseline_is_async else "SYNC")))].copy()
    data["size_str"] = data["size"].astype(str)
    num_gpus = len(data["gpus"].unique())
    
    ##############
    # Plot setup #
    ##############
    
    plt.rcdefaults()
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams['axes.labelpad'] = 5
    plt.rcParams['xtick.major.pad'] = 4.2
    plt.rcParams['ytick.major.pad'] = 0.5
    plt.rcParams['axes.linewidth'] = 1
    
    palette = PALETTE_OG[:num_gpus]
    markers = ["P", "X", "o", "D"][:num_gpus]
    fontsize = 8
    cols = 3
    rows = (len(data_in["benchmark"].unique()) + 1) // cols
    
    fig = plt.figure(figsize=(1.8 * cols + 0.1, 1.5 * rows), dpi=600)
    gs = gridspec.GridSpec(rows, cols)
    plt.subplots_adjust(top=0.9,
                        bottom=0.17,
                        left=0.07,
                        right=0.97,
                        hspace=1,
                        wspace=0.3)

    ##############
    # Main plot  #
    ##############
    
    for b_i, (b, d) in enumerate(data.groupby("benchmark")):
        col = b_i % cols
        row = b_i // cols
        ax = fig.add_subplot(gs[row, col])
        ax = sns.lineplot(x="size_str", y=speedup_column, hue="gpus", data=d, palette=palette, ax=ax, estimator=np.mean,
                          legend=None, ci=99, zorder=2)
        data_averaged = d.groupby(["size_str", "gpus"]).mean()[speedup_column].reset_index()
        ax = sns.scatterplot(x="size_str", y=speedup_column, hue="gpus", style="gpus", data=data_averaged, palette=palette,
                             ax=ax, estimator=np.mean, legend=None, markers=markers, edgecolor="#2f2f2f", zorder=3, size=1, linewidth=0.5)
        plt.xlabel(None)
        plt.ylabel(None)
        
        # Set axis limits;
        x_lim = list(ax.get_xlim())
        ax.set_xlim(x_lim)
        ax.set_ylim((0, 6) if row == 0 else (0, 4))
        
        # Add benchmark name;
        ax.annotate(b, xy=(0.5, 1.05), xycoords="axes fraction", ha="center", color="#2f2f2f", fontsize=fontsize, alpha=1)   
        # Color background to represent linear scaling of performance;
        ax.fill_between(ax.get_xlim(), 0, 1, facecolor="#dddddd", alpha=0.4, zorder=0.4)
        # Grid and axis limits;
        ax.yaxis.grid(True, linewidth=0.5)
        ax.xaxis.grid(False)
        # Line for speedup = 1;
        ax.axhline(y=1, color="0.6", linestyle="--", zorder=1, linewidth=0.5)
        # Ticks;
        ax.yaxis.set_major_locator(plt.LinearLocator(7 if row == 0 else 5))
        ax.set_yticklabels(labels=[f"{l:.1f}x" for l in ax.get_yticks()], ha="right", fontsize=fontsize)
        x_ticks = list(d["size"].unique())
        ax.set_xticks([str(l) for l in x_ticks])
        ax.tick_params(length=2, width=0.5)
        ax.set_xticklabels(labels=[get_exp_label(l, decimal_places=1) for l in x_ticks], ha="center",
                           va="top", rotation=0, fontsize=fontsize - 0.5)
    
        # Add baseline times;
        ax.annotate(f"Baseline {kind} exec. time (ms):", xy=(0, -0.4), fontsize=fontsize - 1, ha="left", xycoords="axes fraction", color="#949494")
        if col == 0:
            ax.annotate(f"{gpu}:", xy=(-0.4, -0.56), fontsize=fontsize - 1, color="#949494", ha="right", xycoords=("data", "axes fraction"))
        for l_i, l in enumerate(x_ticks):
            vals = d[(d["size"] == int(l))][baseline_time_column]
            baseline_mean = np.mean(vals) if len(vals) > 0 else np.nan
            if not math.isnan(baseline_mean):
                ax.annotate(f"{int(1000 * baseline_mean)}", xy=(l_i, -0.56), fontsize=fontsize - 1, color="#2f2f2f", ha="center", xycoords=("data", "axes fraction"))
    
    # Legend;
    labels = ["SYNC, 1 GPU", "2 GPU", "4 GPU", "8 GPU"][:num_gpus]
    patches = [Patch(facecolor=palette[i], edgecolor="#2f2f2f", label=l, linewidth=0.5) for i, l in enumerate(labels)]
    leg = fig.legend(patches, labels, bbox_to_anchor=(0.95, 0.14), fontsize=fontsize,
                     ncol=1, loc="lower right", handlelength=1.2, title=f"{kind}, {gpu},\nvs. ASYNC, 1 GPU",
                     handletextpad=0.2, columnspacing=0.5, title_fontsize=fontsize)
    leg._legend_box.align = "left"
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_facecolor('white')
    
    
def plot_ablation_bars(data_in: pd.DataFrame,
                       gpu: str,
                       gpus: int,
                       speedup_column="speedup",
                       ymax: float=1.6,
                       yticks: int=9,
                       fig: plt.Figure=None,
                       ax: plt.Axes=None,
                       plot_speedup_labels: bool=True):
    
    if fig is None or ax is None:
        plt.rcdefaults()
        sns.set_style("white", {"ytick.left": True, "xtick.bottom": False})
        plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
        plt.rcParams['hatch.linewidth'] = 0.3
        plt.rcParams['axes.labelpad'] = 5
        plt.rcParams['xtick.major.pad'] = 4.2
        plt.rcParams['ytick.major.pad'] = 1
        plt.rcParams['axes.linewidth'] = 0.5
        
        fig = plt.figure(figsize=(2, 0.95), dpi=600)
        gs = gridspec.GridSpec(1, 1)
        plt.subplots_adjust(top=0.78,
                            bottom=0.17,
                            left=0.12,
                            right=0.99,
                            hspace=0.15,
                            wspace=0.8)
        ax = fig.add_subplot(gs[0, 0])
    
    effective_benchmarks = ["MEAN"] + list(data_in["benchmark"].unique())
    fontsize = 4

    data = data_in.copy()
        
    # Compute mean of all benchmarks, grouped by number of GPUs;
    data_mean = data.groupby("policy").mean().reset_index()
    data_mean["benchmark"] = "MEAN"
    new_data = [data_mean]
    new_data += [data]
    data = pd.concat(new_data, ignore_index=True)

    num_policies = len(data["policy"].unique())
    palette = PALETTE_G3[:num_policies]

    ##############
    # Main plot  #
    ##############
    
    ax = sns.barplot(x="benchmark", y=speedup_column, order=effective_benchmarks,
                     hue="policy",
                     palette=palette,
                     data=data,
                     ci=95, capsize=.05, errwidth=0.3, linewidth=0.3,
                     ax=ax, edgecolor="#2f2f2f", estimator=np.mean, saturation=1, zorder=2)
    ax.legend_.remove()  # Hack to remove legend;
    
    ################
    # Refine style #
    ################

    # Grid and axis limits;
    ax.yaxis.grid(True, linewidth=0.4)
    ax.xaxis.grid(False)
    ax.set_xlim((-0.5, len(data["benchmark"].unique()) - 0.5))

    # Axis limits;
    ax.set_ylim((0, ymax))
    # Color background to represent linear scaling of performance;
    ax.fill_between(ax.get_xlim(), 0, 1, facecolor="0.9", alpha=0.5, zorder=0.4, edgecolor="0.9", linewidth=0.1)

    # Ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(yticks))
    ax.set_yticklabels(labels=[f"{l:.1f}x" for l in ax.get_yticks()], ha="right", fontsize=fontsize)
        
    ax.set_xticks([i for i, l in enumerate(data["benchmark"].unique()) if "empty" not in l])
    ax.tick_params(length=2, width=0.5)
    ax.tick_params(axis="x", pad=2)
    ax.set_xticklabels(labels=effective_benchmarks, ha="center", va="top", rotation=0, fontsize=fontsize - 0.2)
        
    # Set "MEAN" labels to a different color;
    for i, l in enumerate(ax.get_xticklabels()):
        if "MEAN" in l._text:
            l.set_color(PALETTE_OG[3])
    # Change color of mean patches, start by filtering the ones with height > 0 and then look for labels with "MEAN";
    try:
        patches_to_color = [a for a in ax.patches if a.get_height() > 0]
        for i, p in enumerate(patches_to_color):
            if i % len(effective_benchmarks) == 0:
                p.set_facecolor(sns.desaturate(p._facecolor, 0.6))
    except IndexError as e:
        print(e)
    # Separate mean with a vertical line;
    ax.axvline(x=0.5, color="0.6", linestyle="--", zorder=1, linewidth=0.3)
    
    plt.ylabel(None)
    plt.xlabel(None)
    
    # Add speedup labels over bars;
    if plot_speedup_labels:
        offsets = []
        for j, g_tmp in data.groupby(["benchmark", "policy"]):
            offsets += [get_upper_ci_size(g_tmp[speedup_column], ci=0.95)]
        offsets = [o if not np.isnan(o) else 0.05 for o in offsets]
        add_labels(ax, vertical_offsets=offsets, rotation=0, format_str="{:.2f}", fontsize=2.2, skip_zero=False)
    
    # Add label with GPU name;
    ax.annotate(f"{gpus} {gpu}s", xy=(0.99, 0.9), xycoords="axes fraction", ha="right", color="#2f2f2f", fontsize=fontsize, alpha=1)   
    
    # Create hierarchical x ticks;
    y_min = -0.03
    y_max = -0.08
    group_labels = effective_benchmarks
    bar_width = ax.patches[0].get_width()  # Get width of a bar
    # labels = ["S" if baseline_is_async else "A", 2, 4, 8][:num_gpus]
    for i in range(len(group_labels)):
        x_start = i - bar_width * (num_policies / 2)
        x_end = i + bar_width * (num_policies / 2)
        x_middle = (x_start + x_end) / 2
        ax.hlines(y_min, x_start, x_end, color="#2f2f2f", linewidth=0.3, clip_on=False, transform=blended_transform_factory(ax.transData, ax.transAxes))
        ax.vlines(x_middle, y_min, y_max, color="#2f2f2f", linewidth=0.3, clip_on=False, transform=blended_transform_factory(ax.transData, ax.transAxes))
        # for l_i, l in enumerate(labels):
        #     start = bar_width * (num_gpus / 2)
        #     ax.annotate(l, xy=(i - start + bar_width / 2 + bar_width * l_i, y_min + 0.02),
        #                 xycoords=blended_transform_factory(ax.transData, ax.transAxes), clip_on=False,
        #                 fontsize=3.2, ha="center")
 
    # Add legend;
    legend_title = "Speedup vs. best policy (MD-Min-Transfer-Time)"
    labels = ["D-Round-Robin", "D-Stream-Aware", "D-Min-Transfer-Time", "MD-Min-Transfer-Time"]
    patches = [Patch(facecolor=palette[i], edgecolor="#2f2f2f", label=l, linewidth=0.5) for i, l in enumerate(labels)]
    leg = fig.legend(patches, labels, bbox_to_anchor=(0.55, 1), fontsize=fontsize,
                     ncol=num_policies, loc="upper center", handlelength=1.2, 
                     handletextpad=0.2, columnspacing=0.5, title=legend_title, title_fontsize=fontsize)
    leg._legend_box.align = "left"
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_facecolor('white')
    
    return ax


#%%###########################
##############################

if __name__ == "__main__":
    
    for g, folder in zip([V100, A100], [V100_RES_FOLDERS, A100_RES_FOLDERS]):
        res_cuda = load_data_cuda_multigpu([os.path.join(g, x) for x in folder], skip_iter=3)
        res_cuda_grouped = res_cuda.groupby(["benchmark", "exec_policy", "gpus"]).mean().dropna().reset_index()
   
        #%% Plot speedup divided by benchmark and number of GPUs;
        plot_speedup_bars(res_cuda_grouped, g)
        save_plot(PLOT_DIR, f"cuda_bars_{g}" + "_{}.{}", date=OUTPUT_DATE, dpi=600)
     
        #%% Plot speedup divided by size, benchmark and number of GPUs;
        plot_speedup_line(res_cuda, g)
        save_plot(PLOT_DIR, f"cuda_lines_{g}" + "_{}.{}", date=OUTPUT_DATE, dpi=600)


