# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:34:37 2021

@author: albyr
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from segretini_matplottini.src.plot_utils import remove_outliers_df_iqr_grouped, compute_speedup_df, save_plot, PALETTE_G3, PALETTE_O, transpose_legend_labels
from load_data import PLOT_DIR

##############################
##############################

INPUT_DATE = "2021_11_02"
OUTPUT_DATE = "2021_11_02"

# V100;
GPU = "V100"
# A100;
GPU = "A100"
SIZE = 20000 # 2048

RES_FOLDER = f"../../../../grcuda-data/results/scheduling_multi_gpu/{GPU}"
SUBFOLDERS = {
    f"{INPUT_DATE}_partition_scaling_b11_low": 4,
    f"{INPUT_DATE}_partition_scaling_b11_high": 12,
    f"{INPUT_DATE}_partition_scaling_b11_veryhigh": 16,
    }

##############################
##############################

def load_data() -> (pd.DataFrame, pd.DataFrame):
    res_folder = os.path.join(RES_FOLDER, f"{INPUT_DATE}_partition_scaling")
    data = []
    for res in os.listdir(res_folder):
        size, gpus, partitions = [int(x) for x in os.path.splitext(res)[0].split("_")]
        try:
            res_data = pd.read_csv(os.path.join(res_folder, res))
            res_data["size"] = size
            res_data["gpus"] = gpus
            res_data["partitions"] = partitions
            data += [res_data]
        except pd._libs.parsers.ParserError as e:
            print(f"error parsing {res}, error={e}")      
    data = pd.concat(data, ignore_index=True)
    # Filter first few iterations;
    data = data[data["num_iter"] > 1]
    # Use only some data size;
    data = data[data["size"] == SIZE]
    # Remove outliers;
    remove_outliers_df_iqr_grouped(data, column="computation_sec", group=["size", "gpus", "partitions"],
                                    reset_index=True, quantile=0.75, drop_index=True, debug=True)
    # Sort data;
    data = data.sort_values(by=["size", "gpus", "partitions", "num_iter"]).reset_index(drop=True)
    # Compute speedups;
    compute_speedup_df(data, key=["size"],
                       baseline_filter_col=["gpus", "partitions"], baseline_filter_val=[1, 1],  
                       speedup_col_name="speedup", time_column="computation_sec",
                       baseline_col_name="baseline_sec", aggregation=np.mean, correction=False)
    # Obtain mean of computation times, grouped;
    data_agg = data.groupby(["size", "gpus", "partitions"]).mean()[["computation_sec", "speedup"]].reset_index()
    return data, data_agg


def load_data_multiconfig(global_speedup: bool=True, remove_outliers: bool=True, main_res_folder: str=RES_FOLDER) -> (pd.DataFrame, pd.DataFrame):
    data = []
    for folder in SUBFOLDERS.keys():
        res_folder = os.path.join(main_res_folder, folder)
        for res in os.listdir(res_folder):
            size, gpus, partitions = [int(x) for x in os.path.splitext(res)[0].split("_")]
            try:
                res_data = pd.read_csv(os.path.join(res_folder, res))
                res_data["size"] = size
                res_data["gpus"] = gpus
                res_data["partitions"] = partitions
                res_data["config"] = SUBFOLDERS[folder]
                data += [res_data]
            except pd._libs.parsers.ParserError as e:
                print(f"error parsing {res}, error={e}")      
                
    data = pd.concat(data, ignore_index=True)
    # Filter first few iterations;
    data = data[data["num_iter"] > 1]
    # Use only some data size;
    data = data[data["size"] == SIZE]
    # Remove outliers;
    if remove_outliers:
        remove_outliers_df_iqr_grouped(data, column="computation_sec", group=["size", "config", "gpus", "partitions" ],
                                        reset_index=True, quantile=0.75, drop_index=True, debug=True)
    # Sort data;
    data = data.sort_values(by=["size", "config", "gpus", "partitions", "num_iter"]).reset_index(drop=True)
    # Compute speedups;
    if global_speedup:
        compute_speedup_df(data, key=["size"],
                           baseline_filter_col=["gpus", "partitions", "config"], baseline_filter_val=[1, 1, min(SUBFOLDERS.values())],  
                           speedup_col_name="speedup", time_column="computation_sec",
                           baseline_col_name="baseline_sec", aggregation=np.mean, correction=False)
    else:
        compute_speedup_df(data, key=["size", "config"],
                           baseline_filter_col=["gpus", "partitions"], baseline_filter_val=[1, 1],  
                           speedup_col_name="speedup", time_column="computation_sec",
                           baseline_col_name="baseline_sec", aggregation=np.mean, correction=False)
    # Obtain mean of computation times, grouped;
    data_agg = data.groupby(["size", "config", "gpus", "partitions"]).mean()[["computation_sec", "speedup"]].reset_index()
    return data, data_agg


def load_data_a100(global_speedup: bool=True, remove_outliers: bool=True, main_res_folder: str=RES_FOLDER) -> (pd.DataFrame, pd.DataFrame):
    data = []
    res_folder = os.path.join(main_res_folder, f"{INPUT_DATE}_partition_scaling")
    for res in os.listdir(res_folder):
        try:
            size, gpus, partitions, config, prefetch = os.path.splitext(res)[0].split("_")
            prefetch = True
        except ValueError:
            size, gpus, partitions, config = os.path.splitext(res)[0].split("_")
            prefetch = False
        size, gpus, partitions, config = [int(x) for x in [size, gpus, partitions, config]]
        try:
            res_data = pd.read_csv(os.path.join(res_folder, res))
            res_data["size"] = size
            res_data["gpus"] = gpus
            res_data["partitions"] = partitions
            res_data["config"] = config
            res_data["prefetch"] = prefetch
            data += [res_data]
        except pd._libs.parsers.ParserError as e:
            print(f"error parsing {res}, error={e}")      
                
    data = pd.concat(data, ignore_index=True)
    # Filter first few iterations;
    data = data[data["num_iter"] > 1]
    # Use only some data size;
    data = data[data["size"] == SIZE]
    # Remove outliers;
    if remove_outliers:
        remove_outliers_df_iqr_grouped(data, column="computation_sec", group=["size", "config", "prefetch", "gpus", "partitions" ],
                                        reset_index=True, quantile=0.75, drop_index=True, debug=True)
    # Sort data;
    data = data.sort_values(by=["size", "prefetch", "config", "gpus", "partitions", "num_iter"]).reset_index(drop=True)
    # Compute speedups;
    if global_speedup:
        compute_speedup_df(data, key=["size", "prefetch"],
                           baseline_filter_col=["gpus", "partitions", "config"], baseline_filter_val=[1, 1, min(data["config"])],  
                           speedup_col_name="speedup", time_column="computation_sec",
                           baseline_col_name="baseline_sec", aggregation=np.mean, correction=False)
    else:
        compute_speedup_df(data, key=["size", "prefetch", "config"],
                           baseline_filter_col=["gpus", "partitions"], baseline_filter_val=[1, 1],  
                           speedup_col_name="speedup", time_column="computation_sec",
                           baseline_col_name="baseline_sec", aggregation=np.mean, correction=False)
    # Obtain mean of computation times, grouped;
    data_agg = data.groupby(["size", "prefetch", "config", "gpus", "partitions"]).mean()[["computation_sec", "speedup"]].reset_index()
    return data, data_agg


def plot_scaling(data_in, skip_low_partition: bool=False, ax=None, fig=None, speedup: bool=False):
    
    # Remove values where the number of partitions is < than the number of GPUs;
    if skip_low_partition:
        data = data_in[data_in["partitions"] >= data_in["gpus"]]
    else:
        data = data_in.copy()

    FONTSIZE = 8
    PALETTE = [PALETTE_G3[i] for i in [1, 2, 3, 4]]
    PALETTE[0] = "#C5E8C5"
    
    new_figure = False
    if fig == None and ax == None:  
        new_figure = True # If true, we are plotting on a new figure;
        plt.rcdefaults()
        sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
        plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
        plt.rcParams['hatch.linewidth'] = 0.3
        plt.rcParams['axes.labelpad'] = 3
        plt.rcParams['xtick.major.pad'] = 4.2
        plt.rcParams['ytick.major.pad'] = 1
        plt.rcParams['axes.linewidth'] = 1
        fig = plt.figure(figsize=(3.5, 2.5), dpi=600)
        plt.subplots_adjust(top=0.95,
                            bottom=0.2,
                            left=0.15,
                            right=0.95)
        
    ax = sns.lineplot(data=data, x="partitions", y="speedup" if speedup else "computation_sec", hue="gpus", ax=ax, legend=False,
                      palette=PALETTE, ci=95, linewidth=0.8)
    # Axes labels;
    plt.xlabel("Number of partitions" if new_figure else None)
    plt.ylabel("Speedup" if speedup else "Exec. time [s]", fontsize=FONTSIZE - 1)
    # Grid and axis limits;
    # ax.set_yscale("log")
    ax.yaxis.grid(True, linewidth=0.3)
    ax.xaxis.grid(False)
    # Axes limits;
    ax.set_xlim((data["partitions"].min(), data["partitions"].max()))
    ax.set_ylim((0.05, 0.10))
    # Ticks;
    ax.tick_params(length=2, width=0.8)
    ax.yaxis.set_major_locator(plt.LinearLocator(6))
    ax.set_yticklabels(labels=[f"{l:.2f}x" if speedup else f"{l:.2f}" for l in ax.get_yticks()], ha="right", fontsize=FONTSIZE - 2)
    x_ticks = [x for x in sorted(list(data["partitions"].unique())) if x % 4 == 0 or x == 1]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels=x_ticks, ha="center", va="top", rotation=0, fontsize=FONTSIZE - 2)
    
    # Title label, if necessary;
    if len(data["gpus"].unique()) > 1 and speedup:
        ax.annotate("vs. 1 A100, P=1", xy=(0.5, 1.03), xycoords="axes fraction", ha="center", color="#2f2f2f", fontsize=FONTSIZE - 1, alpha=1)   
    
    # Legend;
    labels = ["1 A100", "2 A100s", "4 A100s", "8 A100s"]
    patches = [Patch(facecolor=PALETTE[i], edgecolor="#2f2f2f", label=l, linewidth=0.5) for i, l in enumerate(labels)]
    labels, patches = transpose_legend_labels(labels, patches)
    leg = ax.legend(patches, labels, bbox_to_anchor=(1.02, 1.03), fontsize=FONTSIZE - 1, borderpad=0.3,
                    ncol=2, loc="upper right", handlelength=1.2, title=None,
                    handletextpad=0.2, columnspacing=0.5)
    leg._legend_box.align = "left"
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_facecolor('white')
    
    # Add arrow to mark performance gap;
    x_coord = data["partitions"].max() - 0.7
    agg_data = data[data["partitions"] == data["partitions"].max()].groupby("gpus").mean()["computation_sec"]
    plt.annotate(text="", xy=(x_coord, agg_data.min()), xytext=(x_coord, agg_data.max()), annotation_clip=False,
                 arrowprops=dict(arrowstyle="<|-|>", linewidth=0.4, mutation_scale=FONTSIZE / 2, shrinkA=0, capstyle="butt", shrinkB=0, color="#2f2f2f"))
    plt.annotate(text=f"{int(100 * (agg_data.max() - agg_data.min()) / agg_data.min())}%",
                 xy=(x_coord - 0.1, (agg_data.min() + agg_data.max()) / 2), xytext=(0, 0), ha="right", va="center", 
                 textcoords="offset points", annotation_clip=False, fontsize=FONTSIZE - 2)
    return fig, ax
    

def plot_scaling_minmax(data_in_1, data_in_2):
    
    plt.rcdefaults()
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams['axes.labelpad'] = 3
    plt.rcParams['xtick.major.pad'] = 2
    plt.rcParams['ytick.major.pad'] = 1
    plt.rcParams['axes.linewidth'] = 1
    FONTSIZE = 8
    PALETTE = [PALETTE_G3[i] for i in [1, 2, 3, 4]]
    PALETTE[0] = "#C5E8C5"
    fig = plt.figure(figsize=(3.5, 1.5), dpi=600)
    gs = gridspec.GridSpec(1, 2)
    plt.subplots_adjust(top=0.92,
                        bottom=0.2,
                        left=0.15,
                        right=0.95,
                        wspace=0.12)
    
    for i, data_in in enumerate([data_in_1, data_in_2]):
        # Keep just 1 GPU;
        data = data_in[data_in["gpus"] == 1]
        # Remove values where the number of partitions is < than the number of GPUs;
        data = data[data["partitions"] >= data["gpus"]]
        # Remove thr "18" datapoint for consistent scaling;
        data = data[data["partitions"] != 18]

        ax = fig.add_subplot(gs[0, i])
        ax = sns.lineplot(data=data[data["config"] == min(data["config"])], x="partitions", y="speedup", legend=False, color=PALETTE[1], linestyle="--", linewidth=1)
        ax = sns.lineplot(data=data[data["config"] == max(data["config"])], x="partitions", y="speedup", legend=False, color=PALETTE[1], linestyle="--", linewidth=1)
        plt.fill_between(sorted(data["partitions"].unique()),
                         data[data["config"] == min(data["config"])]["speedup"], 
                         data[data["config"] == max(data["config"])]["speedup"],
                         color=PALETTE[0], alpha=0.2)
        # Axes labels;
        plt.xlabel(None)
        if i == 0:
            plt.ylabel("Speedup", fontsize=FONTSIZE)
        else:
            plt.ylabel(None)
        # Grid and axis limits;
        ax.yaxis.grid(True, linewidth=0.5)
        ax.xaxis.grid(False)
        # Axes limits;
        ax.set_xlim((data["partitions"].min(), data["partitions"].max()))
        ax.set_ylim((1.0, 2.5))
        # Ticks;
        ax.tick_params(length=4, width=1)
        ax.yaxis.set_major_locator(plt.LinearLocator(7))
        if i == 0:
            ax.set_yticklabels(labels=[f"{l:.2f}x" for l in ax.get_yticks()], ha="right", fontsize=FONTSIZE - 2)
        else:
            ax.set_yticklabels(labels=[])
        x_ticks = [x for x in sorted(list(data["partitions"].unique())) if x % 4 == 0 or x == 1]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels=x_ticks, ha="center", va="top", rotation=0, fontsize=FONTSIZE - 2)
        
        # Occupancy labels;
        x = sorted(data["partitions"].unique())[-3]
        y1 = data[(data["partitions"] == x) & (data["config"] == min(data["config"]))]["speedup"]
        y2 = data[(data["partitions"] == x) & (data["config"] == max(data["config"]))]["speedup"]
        ax.annotate("Low occupancy", xy=(x, y1), xytext=(-5, -10 if i == 0 else 3), textcoords="offset points", ha="center", color="#2f2f2f", fontsize=FONTSIZE - 1, alpha=1)   
        ax.annotate("High occupancy", xy=(x, y2), xytext=(-5, 3), textcoords="offset points", ha="center", color="#2f2f2f", fontsize=FONTSIZE - 1, alpha=1)   
        # Other labels;
        if i == 0:
            ax.annotate("vs. worst config., P=1", xy=(0.5, 1.03), xycoords="axes fraction", ha="center", color="#2f2f2f", fontsize=FONTSIZE, alpha=1)   
        else:
            ax.annotate("vs. itself, P=1", xy=(0.5, 1.03), xycoords="axes fraction", ha="center", color="#2f2f2f", fontsize=FONTSIZE, alpha=1)   
        ax.annotate("Number of partitions P", xy=(0.55, 0.02), xycoords="figure fraction", ha="center", color="#2f2f2f", fontsize=FONTSIZE, alpha=1)   
    
    return fig, ax


def plot_scaling_minmax_2gpu(data_in, fig=None, ax=None):
    
    FONTSIZE = 8   
    PALETTE = ["#5CCCA7", "#B3767F"]
    
    if fig == None and ax == None:
        plt.rcdefaults()
        sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
        plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
        plt.rcParams['hatch.linewidth'] = 0.3
        plt.rcParams['axes.labelpad'] = 3
        plt.rcParams['xtick.major.pad'] = 2
        plt.rcParams['ytick.major.pad'] = 1
        plt.rcParams['axes.linewidth'] = 0.8
        fig = plt.figure(figsize=(3.5, 1.2), dpi=600)
        gs = gridspec.GridSpec(1, 2)
        plt.subplots_adjust(top=0.92,
                            bottom=0.2,
                            left=0.12,
                            right=0.95,
                            wspace=0.12)
        # Plot gap between low and high occupancy on the 2 GPUs;
        ax = fig.add_subplot(gs[0, 0])
    
    for i, (gpu, d) in enumerate(data_in.groupby("gpu", sort=False)):
        # Keep just 1 GPU;
        d = d[d["gpus"] == 1]
        # Remove values where the number of partitions is < than the number of GPUs;
        d = d[d["partitions"] >= d["gpus"]]
        
        ax = sns.lineplot(data=d[d["config"] == min(d["config"])], x="partitions", y="computation_sec", legend=False, color=PALETTE[i], linestyle="--", linewidth=0.7)
        ax = sns.lineplot(data=d[d["config"] == max(d["config"])], x="partitions", y="computation_sec", legend=False, color=PALETTE[i], linestyle="--", linewidth=0.7)

        plt.fill_between(sorted(d["partitions"].unique()),
                         d[d["config"] == min(d["config"])]["computation_sec"], 
                         d[d["config"] == max(d["config"])]["computation_sec"],
                         color=PALETTE[i], alpha=0.2)
        # Axes labels;
        plt.xlabel(None)
        plt.ylabel("Exec. time, log-scale [s]", fontsize=FONTSIZE - 1)
        # Grid and axis limits;
        ax.yaxis.grid(True, linewidth=0.3)
        ax.xaxis.grid(False)
        # Axes limits;
        ax.set_xlim((d["partitions"].min(), d["partitions"].max()))
        ax.set_ylim((0.05, 0.3))
        # Ticks;
        ax.tick_params(length=2, width=0.8)
        x_ticks = [x for x in sorted(list(d["partitions"].unique())) if x % 4 == 0 or x == 1]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels=x_ticks, ha="center", va="top", rotation=0, fontsize=FONTSIZE - 2)
        
        # Occupancy labels;
        x = sorted(d["partitions"].unique())[1]
        y1 = d[(d["partitions"] == x) & (d["config"] == min(d["config"]))]["computation_sec"]
        y2 = d[(d["partitions"] == x) & (d["config"] == max(d["config"]))]["computation_sec"]
        ax.annotate("Low occupancy", xy=(x, y1), xytext=(6, -6 if i == 0 else -2), textcoords="offset points", ha="left", color="#2f2f2f", fontsize=FONTSIZE - 3, alpha=0.8)   
        ax.annotate("High occupancy", xy=(x, y2), xytext=(6, -8 if i == 0 else -6), textcoords="offset points", ha="left", color="#2f2f2f", fontsize=FONTSIZE - 3, alpha=0.8)   
        # Other labels;
        # ax.annotate("vs. worst config., P=1", xy=(0.5, 1.03), xycoords="axes fraction", ha="center", color="#2f2f2f", fontsize=FONTSIZE - 1, alpha=1)   
        ax.annotate("Number of partitions P", xy=(0.55, 0.02), xycoords="figure fraction", ha="center", color="#2f2f2f", fontsize=FONTSIZE - 1, alpha=1)   
    
    # Set scale to logarithmic on the y axis, update tick labels;
    ax.set_yscale("log")   
    ax.yaxis.set_major_locator(plt.LinearLocator(6))
    ax.set_yticklabels(labels=[f"{l:.2f}" for l in ax.get_yticks()], ha="right", fontsize=FONTSIZE - 2)
    ax.yaxis.set_minor_formatter(NullFormatter())  # Hide any weird log tick label
    
    for i, (gpu, d) in enumerate(data_in.groupby("gpu", sort=False)):
        # Keep just 1 GPU;
        d = d[d["gpus"] == 1]
        # Remove values where the number of partitions is < than the number of GPUs;
        d = d[d["partitions"] >= d["gpus"]]
        # Add arrow to mark performance gap;
        x_coord = d["partitions"].max() - 0.7
        low = d[(d["config"] == min(d["config"])) & (d["partitions"] == d["partitions"].max())]["computation_sec"].iloc[0]
        high = d[(d["config"] == max(d["config"])) & (d["partitions"] == d["partitions"].max())]["computation_sec"].iloc[0]
        y_diff = np.abs(high - low) / min([high, low])
        y_coord =  (high + low) / 2 
        ax.annotate(text=f"{100 * y_diff:.2f}%",
                    xy=(x_coord, y_coord), xytext=(0, -8), textcoords="offset points",
                    ha="right", va="center",  fontsize=FONTSIZE - 2,
                    arrowprops=dict(arrowstyle="-|>", linewidth=0.4, mutation_scale=FONTSIZE / 2, shrinkA=2, capstyle="butt", shrinkB=0, color="#2f2f2f"),
                    bbox=dict(boxstyle="square,pad=0.05", fc="w", alpha=0, ec=None))
        
    # Legend;
    labels = list(data_in["gpu"].unique())
    patches = [Patch(facecolor=PALETTE[i], edgecolor="#2f2f2f", label=l, linewidth=0.5) for i, l in enumerate(labels)]
    leg = ax.legend(patches, labels, bbox_to_anchor=(1.02, 1.03), fontsize=FONTSIZE - 1, borderpad=0.3,
                      ncol=1, loc="upper right", handlelength=1.2, title=None,
                      handletextpad=0.2, columnspacing=0.5)
    leg._legend_box.align = "left"
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_facecolor('white')

    return fig, ax


def v100_vs_a100_and_multigpu_plot(data_agg, data_a100):
    # Setup plot;
    plt.rcdefaults()
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams['axes.labelpad'] = 1
    plt.rcParams['xtick.major.pad'] = 2
    plt.rcParams['ytick.major.pad'] = 1
    plt.rcParams['axes.linewidth'] = 0.8
    
    fig = plt.figure(figsize=(3.5, 1.2), dpi=600)
    gs = gridspec.GridSpec(1, 2)
    plt.subplots_adjust(top=0.95,
                        bottom=0.2,
                        left=0.10,
                        right=0.97,
                        wspace=0.3)
     
    # Plot gap between low and high occupancy on the 2 GPUs;
    fig, ax = plot_scaling_minmax_2gpu(data_agg, fig=fig, ax=fig.add_subplot(gs[0, 0]))
    # Plot multi-GPU scaling of A100;
    fig, ax = plot_scaling(data_a100[data_a100["prefetch"] == True], ax=fig.add_subplot(gs[0, 1]), fig=fig, speedup=False) 
    return fig, ax

#%%###########################
##############################

if __name__ == "__main__":
    
    # if GPU == "V100":
    #     # # Plot;
    #     # data, data_agg = load_data()
    #     # fig, ax = plot_scaling(data)    
    #     # save_plot(PLOT_DIR, "cuda_partition_scaling" + "_{}.{}", date=OUTPUT_DATE, dpi=600)
        
    #     # Second plot;
    #     _, data_agg_1 = load_data_multiconfig(True)
    #     _, data_agg_2 = load_data_multiconfig(False)
    #     fig, ax = plot_scaling_minmax(data_agg_1, data_agg_2)    
    #     save_plot(PLOT_DIR, "cuda_partition_scaling_minmax" + "_{}.{}", date=OUTPUT_DATE, dpi=600)
    # elif GPU == "A100":
    #     data, data_agg = load_data_a100(global_speedup=True, remove_outliers=False)
    #     for i, g in data.groupby("config"):
    #         fig, ax = plot_scaling(g[g["prefetch"] == True]) 
            
    #%% Compare V100 and A100;
    _, data_agg_v100 = load_data_multiconfig(True, remove_outliers=False, main_res_folder="../../../../grcuda-data/results/scheduling_multi_gpu/V100")
    data_agg_v100["prefetch"] = True
    # Remove thr "18" datapoint for consistent scaling;
    data_agg_v100 = data_agg_v100[data_agg_v100["partitions"] != 18]
    data_a100, data_agg_a100 = load_data_a100(global_speedup=True, remove_outliers=False, main_res_folder="../../../../grcuda-data/results/scheduling_multi_gpu/A100")
    data_agg_a100 = data_agg_a100[data_agg_a100["prefetch"] == True]
    # Concatenate results;
    data_agg_v100["gpu"] = "V100"
    data_agg_a100["gpu"] = "A100"
    data_agg = pd.concat([data_agg_v100, data_agg_a100], ignore_index=True)
    fig, ax = v100_vs_a100_and_multigpu_plot(data_agg, data_a100)    
    save_plot(PLOT_DIR, "cuda_partition_scaling_minmax_2gpu" + "_{}.{}", date=OUTPUT_DATE, dpi=600)
    