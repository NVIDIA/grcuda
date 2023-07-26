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
from segretini_matplottini.src.plot_utils import COLORS, get_exp_label, get_ci_size, save_plot


DEFAULT_RES_DIR = "../../../../grcuda-data/results/scheduling_nvprof_log"

# 960
INPUT_DATE_960 = "2020_10_07_960"
# P100
INPUT_DATE_P100 = "2020_10_10_P100"
# 1660
INPUT_DATE_1660 = "2020_10_10_1660"

OUTPUT_DATE = "2020_10_11"
PLOT_DIR = "../../../../grcuda-data/plots"

BENCHMARK_NAMES = {"b1": "Vector Squares", "b5": "B&S",  "b8": "Images",  "b6": "ML Ensemble", "b7": "HITS","b10": "DL"}

LABEL_DICT = {"ct_overlap_perc": "CT", "tc_overlap_perc": "TC", "cc_overlap_perc": "CC", "total_overlap_perc": "TOT", "fake_perc": ""}
LABEL_LEGEND_DICT = {"ct_overlap_perc": "CT, computation w.r.t transfer",
                     "tc_overlap_perc": "TC, transfer w.r.t computation",                 
                     "cc_overlap_perc": "CC, computation w.r.t computation",
                     "total_overlap_perc": "TOT, any type of overlap"
                     }

SPEEDUPS = {
    "b1": 1.17,
    "b5": 1.33,
    "b6": 1.22,
    "b7": 1.13,
    "b8": 1.32,
    "b10": 1.34,
    }

SPEEDUPS_960 = {
    "b1": 1.17,
    "b5": 1.33,
    "b6": 1.22,
    "b7": 1.13,
    "b8": 1.55,
    "b10": 1.34,
    }

SPEEDUPS_P100 = {
    "b1": 2.55,
    "b5": 2.79,
    "b6": 1.39,
    "b7": 1.33,
    "b8": 1.49,
    "b10": 1.17,
    }

SPEEDUPS_1660 = {
    "b1": 2.68,
    "b5": 1.83,
    "b6": 1.28,
    "b7": 1.38,
    "b8": 1.34,
    "b10": 1.19,
    }

GPU_NAMES = ["GTX 960", "GTX 1660 Super", "Tesla P100"]

#%%
if __name__ == "__main__": 
    
    # data = pd.read_csv(os.path.join(DEFAULT_RES_DIR, INPUT_DATE_P100, "summary.csv"))
        
    # # Add a fake column for visualization;
    # data["fake_perc"] = 0.0
    # data["benchmark_num"] = [list(BENCHMARK_NAMES.keys()).index(x) for x in data["benchmark"]]

    # # Pivot the dataset;
    # data_pivot = pd.melt(data, id_vars=[data.columns[0], data.columns[-1]], value_vars=data.columns[1:-1],
    #     var_name="overlap_type", value_name="overlap_perc")
    # data_pivot = data_pivot.sort_values(["benchmark_num"], ignore_index=True, kind="mergesort")
    
    # # Remove the fake column for the last benchmark;
    # last_b = data_pivot["benchmark"].unique()[-1]
    # data_pivot = data_pivot[~((data_pivot["benchmark"] == last_b) & (data_pivot["overlap_type"] == "fake_perc"))]
    
    # # Obtain x values for the plot;
    # x = np.arange(len(data_pivot))
    # # Obtain labels;
    # x_labels = [LABEL_DICT[l] for l in data_pivot["overlap_type"]]
    # # Obtain y;
    # y = data_pivot["overlap_perc"]
    
    # sns.set_style("white", {"ytick.left": True})
    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # plt.rcParams['axes.titlepad'] = 25 
    # plt.rcParams['axes.labelpad'] = 9 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    # plt.rcParams['xtick.major.pad'] = 1 
    
    # num_col = len(data_pivot["benchmark"].unique())
    # # fig = plt.figure(figsize=(1.2 * num_col, 3))
    # # gs = gridspec.GridSpec(1, num_col)
    
    # fig = plt.figure(figsize=(1.2 * num_col, 2.8)) 
    # ax = fig.add_subplot()
    # plt.subplots_adjust(top=0.72,
    #                 bottom=0.25,
    #                 left=0.08,
    #                 right=.99,
    #                 hspace=0.9,
    #                 wspace=0.0)
    # p = [COLORS["b3"], COLORS["b8"], COLORS["y3"], COLORS["r5"], COLORS["bb4"], COLORS["bb5"]]
    # # p = ["#FFEDAB", "#FFDB8C", "#FFC773", "#FFAF66"]
    # p = ["#C8FCB6", "#96DE9B", "#66B784", "#469E7B"]
    # palette = (p[:len(LABEL_DICT) - 1] + ["#ffffff"]) * num_col
    # palette = palette[:len(x)]
    # edgecolor = ["#ffffff" if (p == "#ffffff" or y[i] <= 0) else "#2f2f2f" for i, p in enumerate(palette)]
    
    # bar_width = 0.8
    
    # white_bars = (([1] * len(LABEL_LEGEND_DICT) + [0]) * num_col)[:-1]
    # edgecolor_white_bars = ["#ffffff" if p == "#ffffff" else "#0f0f0f" for i, p in enumerate(palette)]
    # ax.bar(x, white_bars, bar_width, color="0.8", edgecolor=edgecolor_white_bars, alpha=0.5)
    # ax.bar(x, y, bar_width, color=palette, edgecolor=edgecolor)
    # ax.set_xticks(x)
    # ax.set_xticklabels(x_labels, fontsize=9)
    
    # ax.set_xlim((0 - bar_width / 2 - 0.2, len(x) - 1 + bar_width / 2 + 0.2))
    # ax.set_ylim((0, 1))
    # # Set the y ticks;
    # ax.yaxis.set_major_locator(plt.LinearLocator(6))
    # ax.set_yticklabels(labels=[f"{int(l * 100)}%" for l in ax.get_yticks()], ha="right", fontsize=11)
    # ax.grid(True, axis="y")
    
    # # Add benchmark name;
    # x_label_pos = 1 / (2 * len(BENCHMARK_NAMES))
    # def get_x_label_pos(i):
    #     base_pos = 2 * x_label_pos * i + x_label_pos
    #     if i == 0:
    #         return base_pos - 0.015
    #     elif i == len(BENCHMARK_NAMES) - 1:
    #         return base_pos + 0.015
    #     else:
    #         return base_pos
    # for i, b in enumerate(BENCHMARK_NAMES):
    #     ax.annotate(f"{BENCHMARK_NAMES[b]}", xy=(get_x_label_pos(i), -0.28), fontsize=12, ha="center", xycoords="axes fraction")
    #     ax.annotate(f"Speedup: ", xy=(get_x_label_pos(i) - 0.02, -0.43), fontsize=10, ha="center", xycoords="axes fraction")
    #     ax.annotate(f"{SPEEDUPS[b]:.2f}x", xy=(get_x_label_pos(i) + 0.045, -0.43), fontsize=10, ha="center", xycoords="axes fraction", color="#469E7B")
        
    # # Legend;  
    # labels = [LABEL_LEGEND_DICT[l] for l in list(LABEL_DICT.keys())[:-1]]
    # custom_lines = [Patch(facecolor=palette[i], edgecolor="#2f2f2f", label=l)
    #                 for i, l in enumerate(labels)]
    # leg = fig.legend(custom_lines, labels, bbox_to_anchor=(1, 1), fontsize=10, ncol=1)
    # leg.set_title("Type of overlap")
    # leg._legend_box.align = "left"
    # leg.get_frame().set_facecolor('white')
    
    # plt.suptitle("Amount of transfer and computaton\noverlap for each benchmark", fontsize=16, x=.05, y=0.95, ha="left")

    # save_plot(PLOT_DIR, "overlap_{}.{}", OUTPUT_DATE)
    
    
    # %% Plot both GPUs;
    
    data_p100 = pd.read_csv(os.path.join(DEFAULT_RES_DIR, INPUT_DATE_P100, "summary.csv"))
    data_960 = pd.read_csv(os.path.join(DEFAULT_RES_DIR, INPUT_DATE_960, "summary.csv"))
    data_1660 = pd.read_csv(os.path.join(DEFAULT_RES_DIR, INPUT_DATE_1660, "summary.csv"))
    data_list = [data_960, data_1660, data_p100]
    speedups = [SPEEDUPS_960, SPEEDUPS_1660, SPEEDUPS_P100]
    
    sns.set_style("white", {"ytick.left": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 25 
    plt.rcParams['axes.labelpad'] = 9 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    plt.rcParams['xtick.major.pad'] = 1 
    
    num_col = len(data_p100["benchmark"].unique())
    num_row = len(data_list)
    fig = plt.figure(figsize=(1.2 * num_col, 2.1 * num_row))
    gs = gridspec.GridSpec(len(data_list), 1)
    
    plt.subplots_adjust(top=0.77,
                    bottom=0.09,
                    left=0.08,
                    right=.99,
                    hspace=0.8,
                    wspace=0.0)
    p = [COLORS["b3"], COLORS["b8"], COLORS["y3"], COLORS["r5"], COLORS["bb4"], COLORS["bb5"]]
    # p = ["#FFEDAB", "#FFDB8C", "#FFC773", "#FFAF66"]
    p = ["#C8FCB6", "#96DE9B", "#66B784", "#469E7B"]
    palette = (p[:len(LABEL_DICT) - 1] + ["#ffffff"]) * num_col
    # palette = palette[:len(x)]
    
    bar_width = 0.8
    
    for i, data in enumerate(data_list):
        
        ax = fig.add_subplot(gs[i, 0])
        
        # Add a fake column for visualization;
        data["fake_perc"] = 0.0
        data["benchmark_num"] = [list(BENCHMARK_NAMES.keys()).index(x) for x in data["benchmark"]]
    
        # Pivot the dataset;
        data_pivot = pd.melt(data, id_vars=[data.columns[0], data.columns[-1]], value_vars=data.columns[1:-1],
            var_name="overlap_type", value_name="overlap_perc")
        data_pivot = data_pivot.sort_values(["benchmark_num"], ignore_index=True, kind="mergesort")
        
        # Remove the fake column for the last benchmark;
        last_b = data_pivot["benchmark"].unique()[-1]
        data_pivot = data_pivot[~((data_pivot["benchmark"] == last_b) & (data_pivot["overlap_type"] == "fake_perc"))]

        # Obtain x values for the plot;
        x = np.arange(len(data_pivot))
        # Obtain labels;
        x_labels = [LABEL_DICT[l] for l in data_pivot["overlap_type"]]
        # Obtain y;
        y = data_pivot["overlap_perc"]
        edgecolor = ["#ffffff" if (p == "#ffffff" or y[j] <= 0) else "#2f2f2f" for j, p in enumerate(palette)]

        white_bars = (([1] * len(LABEL_LEGEND_DICT) + [0]) * num_col)[:-1]
        edgecolor_white_bars = ["#ffffff" if p == "#ffffff" else "#0f0f0f" for j, p in enumerate(palette)]
        ax.bar(x, white_bars, bar_width, color="0.8", edgecolor=edgecolor_white_bars, alpha=0.5)
        ax.bar(x, y, bar_width, color=palette, edgecolor=edgecolor)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        
        ax.set_xlim((0 - bar_width / 2 - 0.2, len(x) - 1 + bar_width / 2 + 0.2))
        ax.set_ylim((0, 1))
        # Set the y ticks;
        ax.yaxis.set_major_locator(plt.LinearLocator(6))
        ax.set_yticklabels(labels=[f"{int(l * 100)}%" for l in ax.get_yticks()], ha="right", fontsize=11)
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
        ax.annotate(f"{GPU_NAMES[i]}", xy=(-0.065, 1.35 if i == 0 else 1.18), fontsize=16, ha="left", xycoords="axes fraction")
        for j, b in enumerate(BENCHMARK_NAMES):
            if i == 0:
                ax.annotate(f"{BENCHMARK_NAMES[b]}", xy=(get_x_label_pos(j), 1.1), fontsize=12, ha="center", xycoords="axes fraction")
            ax.annotate("Speedup: ", xy=(get_x_label_pos(j) - 0.02, -0.35), fontsize=10, ha="center", xycoords="axes fraction")
            ax.annotate(f"{speedups[i][b]:.2f}x", xy=(get_x_label_pos(j) + 0.045, -0.35), fontsize=10, ha="center", xycoords="axes fraction", color="#469E7B")
            
    # Legend;  
    labels = [LABEL_LEGEND_DICT[l] for l in list(LABEL_DICT.keys())[:-1]]
    custom_lines = [Patch(facecolor=palette[i], edgecolor="#2f2f2f", label=l)
                    for i, l in enumerate(labels)]
    leg = fig.legend(custom_lines, labels, bbox_to_anchor=(1, 1), fontsize=10, ncol=1)
    leg.set_title("Type of overlap")
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    plt.suptitle("Amount of transfer and computaton\noverlap for each benchmark", fontsize=16, x=.02, y=0.98, ha="left")

    save_plot(PLOT_DIR, "overlap_full_{}.{}", OUTPUT_DATE)
    