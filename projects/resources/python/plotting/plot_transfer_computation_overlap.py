#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:45:50 2020

@author: alberto.parravicini
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy.stats.mstats import gmean
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.lines as lines
from plot_utils import COLORS, get_exp_label, get_ci_size, save_plot


DEFAULT_RES_DIR = "../../../../data/nvprof_log"

INPUT_DATE = "2020_07_20"
OUTPUT_DATE = "2020_07_20"
PLOT_DIR = "../../../../data/plots"

BENCHMARK_NAMES = {"b1": "Vector Squares", "b5": "B&S", "b6": "ML Ensemble", "b7": "HITS", "b8": "Images"}

LABEL_DICT = {"ct_overlap_perc": "CT", "tc_overlap_perc": "TC", "cc_overlap_perc": "CC", "total_overlap_perc": "TOT", "fake_perc": ""}
LABEL_LEGEND_DICT = {"ct_overlap_perc": "CT, computation w.r.t transfer",
                     "tc_overlap_perc": "TC, transfer w.r.t computation",                 
                     "cc_overlap_perc": "CC, computation w.r.t computation",
                     "total_overlap_perc": "TOT, any type of overlap"
                     }

SPEEDUPS = {
    "b1": 1.17,
    "b5": 1.34,
    "b6": 1.25,
    "b7": 1.14,
    "b8": 1.30,
    }

if __name__ == "__main__":
    data = pd.read_csv(os.path.join(DEFAULT_RES_DIR, INPUT_DATE, "summary.csv"))
        
    # Add a fake column for visualization;
    data["fake_perc"] = 0.0
    
    # Pivot the dataset;
    data_pivot = pd.melt(data, id_vars=data.columns[0], value_vars=data.columns[1:],
        var_name="overlap_type", value_name="overlap_perc")
    data_pivot = data_pivot.sort_values(list(data_pivot.columns[:1]), ignore_index=True, kind="mergesort")
    
    # Remove the fake column for the last benchmark;
    last_b = data_pivot["benchmark"].unique()[-1]
    data_pivot = data_pivot[~((data_pivot["benchmark"] == last_b) & (data_pivot["overlap_type"] == "fake_perc"))]
    
    # Obtain x values for the plot;
    x = np.arange(len(data_pivot))
    # Obtain labels;
    x_labels = [LABEL_DICT[l] for l in data_pivot["overlap_type"]]
    # Obtain y;
    y = data_pivot["overlap_perc"]
    
    sns.set_style("white", {"ytick.left": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 25 
    plt.rcParams['axes.labelpad'] = 9 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    plt.rcParams['xtick.major.pad'] = 1 
    
    num_col = len(data_pivot["benchmark"].unique())
    # fig = plt.figure(figsize=(1.2 * num_col, 3))
    # gs = gridspec.GridSpec(1, num_col)
    
    fig = plt.figure(figsize=(1.2 * num_col, 2.5)) 
    ax = fig.add_subplot()
    plt.subplots_adjust(top=0.75,
                    bottom=0.25,
                    left=0.08,
                    right=.99,
                    hspace=0.9,
                    wspace=0.0)
    p = [COLORS["b3"], COLORS["b8"], COLORS["y3"], COLORS["r5"], COLORS["bb4"], COLORS["bb5"]]
    # p = ["#FFEDAB", "#FFDB8C", "#FFC773", "#FFAF66"]
    p = ["#C8FCB6", "#96DE9B", "#66B784", "#469E7B"]
    palette = (p[:len(LABEL_DICT) - 1] + ["#ffffff"]) * num_col
    palette = palette[:len(x)]
    edgecolor = ["#ffffff" if (p == "#ffffff" or y[i] <= 0) else "#2f2f2f" for i, p in enumerate(palette)]
    
    bar_width = 0.8
    
    white_bars = (([1] * len(LABEL_LEGEND_DICT) + [0]) * num_col)[:-1]
    edgecolor_white_bars = ["#ffffff" if p == "#ffffff" else "#0f0f0f" for i, p in enumerate(palette)]
    ax.bar(x, white_bars, bar_width, color="0.8", edgecolor=edgecolor_white_bars, alpha=0.5)
    ax.bar(x, y, bar_width, color=palette, edgecolor=edgecolor)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    
    ax.set_xlim((0 - bar_width / 2 - 0.2, len(x) - 1 + bar_width / 2 + 0.2))
    ax.set_ylim((0, 1))
    # Set the y ticks;
    ax.yaxis.set_major_locator(plt.LinearLocator(6))
    ax.set_yticklabels(labels=[f"{int(l * 100)}%" for l in ax.get_yticks()], ha="right", fontsize=9)
    ax.grid(True, axis="y")
    
    # Add benchmark name;
    x_label_pos = 1 / (2 * len(BENCHMARK_NAMES))
    def get_x_label_pos(i):
        base_pos = 2 * x_label_pos * i + x_label_pos
        if i == 0:
            return base_pos - 0.015
        elif i == len(BENCHMARK_NAMES) - 1:
            return base_pos + 0.015
        else:
            return base_pos
    for i, b in enumerate(BENCHMARK_NAMES):
        ax.annotate(f"{BENCHMARK_NAMES[b]}", xy=(get_x_label_pos(i), -0.25), fontsize=10, ha="center", xycoords="axes fraction")
        ax.annotate(f"Speedup: ", xy=(get_x_label_pos(i) - 0.02, -0.4), fontsize=8, ha="center", xycoords="axes fraction")
        ax.annotate(f"{SPEEDUPS[b]:.2f}x", xy=(get_x_label_pos(i) + 0.045, -0.4), fontsize=8, ha="center", xycoords="axes fraction", color="#469E7B")
        
    # Legend;  
    labels = [LABEL_LEGEND_DICT[l] for l in list(LABEL_DICT.keys())[:-1]]
    custom_lines = [Patch(facecolor=palette[i], edgecolor="#2f2f2f", label=l)
                    for i, l in enumerate(labels)]
    leg = fig.legend(custom_lines, labels, bbox_to_anchor=(0.98, 1), fontsize=8, ncol=1)
    leg.set_title("Type of overlap")
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    plt.suptitle("Amount of overlap for each benchmark", fontsize=12, x=.05, y=0.95, ha="left")

    save_plot(PLOT_DIR, "overlap_{}.{}", OUTPUT_DATE)