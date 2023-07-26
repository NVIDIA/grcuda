# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:00:58 2022

@author: albyr
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from segretini_matplottini.src.plot_utils import COLORS, get_exp_label, get_ci_size, save_plot
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
from load_data import PLOT_DIR

OUTPUT_DATE = "2022_01_19"

DEFAULT_RES_DIR = "../../connection_graph/datasets"
NUM_GPU = 8
DATASET = "connection_graph_{}_{}.csv"
V100 = "V100"
V100_DATA = DATASET.format(NUM_GPU, V100.lower())
A100 = "A100"
A100_DATA = DATASET.format(NUM_GPU, A100.lower())

##############################
##############################

def plot_heatmap(data_dict: dict) -> plt.Figure:
    
    def plot_gpu(data: pd.DataFrame, ax_gpu: plt.Axes, column: int, ax_cbar: plt.Axes=None):
        # Do not plot CPU, we plot it separetely;
        data_gpu = data[data.index != "CPU"]
        # Mask the lower anti-triagonal matrix, excluding the main diagonal, so it's not shown;
        mask = np.zeros_like(data_gpu)
        mask[np.tril_indices_from(mask)] = True
        mask ^= np.eye(NUM_GPU).astype(bool)

        # Main heatmap plot;
        ax_gpu = sns.heatmap(data_gpu, square=True, mask=mask, vmin=0, vmax=max_bandwidth, linewidth=LINEWIDTH, 
                             linecolor=linecolors, cmap=custom_cm, ax=ax_gpu, cbar_ax=ax_cbar, cbar=ax_cbar is not None,
                             cbar_kws={"ticks": [0] + sorted_steps_colorbar})
        # Add hatches to the main diagonal (https://stackoverflow.com/questions/55285013/adding-hatches-to-seaborn-heatmap-plot);
        x = np.arange(len(data_gpu.columns) + 1)
        y = np.arange(len(data_gpu.index) + 1)
        zm = np.ma.masked_less(data_gpu.values, 200)
        ax_gpu.pcolor(x, y, zm, hatch="//" * 3, alpha=0.0)
        
        # Add borders to the plot;
        sns.despine(ax=ax_gpu, top=False, right=False)
        # Hide axis labels;
        ax_gpu.set(xlabel=None, ylabel=None)
        # Hide tick labels;
        ax_gpu.tick_params(labelbottom=False, top=False, labelsize=FONTSIZE, pad=2)   
        ax_gpu.set_yticks([i + 0.5 for i in range(NUM_GPU)])
        ax_gpu.set_yticklabels([f"GPU{i}" for i in range(NUM_GPU)])
        
        # Dotted lines from left to main diagonal;
        for i in range(1, NUM_GPU):
            ax_gpu.axhline(i + 0.5, xmin=0, xmax=i / NUM_GPU, color="#2f2f2f", linewidth=1, linestyle=":")
        
        # Customize colorbar;
        if ax_cbar is not None:
            # Add border around colorbar;
            cbar = ax_gpu.collections[0].colorbar
            for spine in cbar.ax.spines.values():
                spine.set(visible=True, linewidth=LINEWIDTH, edgecolor="black")
            # Customize labels of colorbar
            cbar.ax.set_yticklabels([f"{x}" for x in [0] + sorted_steps_colorbar]) 
            cbar.ax.tick_params(labelsize=FONTSIZE, pad=1, size=2)
            cbar.ax.annotate("GB/s", xy=(0, 1.02), fontsize=FONTSIZE, ha="left", xycoords="axes fraction", color="#2f2f2f")
        return ax_gpu, ax_cbar
        
    def plot_cpu(data: pd.DataFrame, ax: plt.Axes, column: int):
        # Draw the heatmap for the CPU;
        data_cpu = data[data.index == "CPU"]
        ax_cpu = sns.heatmap(data_cpu, square=True, vmin=0, vmax=max_bandwidth, linewidth=LINEWIDTH, 
                             linecolor=linecolors, cmap=custom_cm, ax=ax, cbar=False)
        # Put x-tick labels on top;
        ax_cpu.xaxis.tick_top()
        ax_cpu.xaxis.set_label_position("top")
        # Hide axis labels;
        ax_cpu.set(xlabel=None, ylabel=None)
        # Show x-tick labels;
        ax_cpu.set_yticks([0.5])
        ax_cpu.set_yticklabels(["CPU"])
        ax_cpu.set_xticks([i + 0.5 for i in range(NUM_GPU)])
        ax_cpu.set_xticklabels([f"G{i}" for i in range(NUM_GPU)])
        ax_cpu.tick_params(labeltop=True, top=True, pad=0.1, labelsize=FONTSIZE)    
        
        # Draw lines between CPU heatmap and GPU heatmap;
        for i in range(0, NUM_GPU):
            ax_cpu.axvline(i + 0.5, ymin=0, ymax=-2, color="#2f2f2f", linewidth=1, linestyle=":", clip_on=False)
            
        # Add tree above GPUs to show CPU interconnection;
        base = 1.7
        for i in range(0, NUM_GPU):
            ax_cpu.axvline(i + 0.5, ymin=base, ymax=base + 0.3, color="#2f2f2f", linewidth=0.5, linestyle="-", clip_on=False)
        for i in range(0, NUM_GPU, 2):
            ax_cpu.axhline(-1, xmin=(i + 0.5) / NUM_GPU, xmax=(i + 1.5) / NUM_GPU, color="#2f2f2f", linewidth=0.5, linestyle="-", clip_on=False, zorder=89)
        for i in range(0, NUM_GPU, 2):
            ax_cpu.axvline(i + 1, ymin=base + 0.3, ymax=base + 0.6, color="#2f2f2f", linewidth=0.5, linestyle="-", clip_on=False)   
        for i in range(0, NUM_GPU, 4):
            ax_cpu.axhline(-1.3, xmin=(i + 1) / NUM_GPU, xmax=(i + 3) / NUM_GPU, color="#2f2f2f", linewidth=0.5, linestyle="-", clip_on=False, zorder=89)
            ax.annotate(f"PCIe Tree {i // 4}", xy=((i + 2) / NUM_GPU, 2.4), fontsize=FONTSIZE, ha="center", color="#2f2f2f", clip_on=False, xycoords="axes fraction")
        return ax_cpu
    
    #################
    # Preprocessing #
    #################
    
    # Obtain maximum of the matrix, excluding the main diagonal;
    max_bandwidth = 0
    for d in data_dict.values():
        data_tmp = d[d.index != "CPU"]
        max_bandwidth = max(max_bandwidth, (data_tmp.to_numpy() - np.eye(NUM_GPU) * data_tmp.to_numpy().diagonal()).max())
    
    # Black outline for non-empty cells, else white;
    linecolors = ["#2f2f2f" if i <= j else (0, 0, 0, 0) for i in range(NUM_GPU) for j in range(NUM_GPU)]
    # Black and white colormap, from black to white (https://stackoverflow.com/questions/58597226/how-to-customize-the-colorbar-of-a-heatmap);
    num_colors = 200
    cm = LinearSegmentedColormap.from_list("gray-custom", ["0.2", "white"], N=num_colors)
    custom_colors = np.array([list(cm(i)) for i in np.linspace(0, 1, num_colors)])
    # Add discrete steps, including the CPU;
    values = set()
    for d in data_dict.values():
        values = values.union(set(d.to_numpy().reshape(-1)))
    sorted_steps_colorbar = sorted([c for c in values if c <= max_bandwidth], reverse=True)
    for c in sorted_steps_colorbar:
        custom_colors[:int(num_colors * c / max_bandwidth) + 1, :] = cm(c / max_bandwidth)
    custom_cm = ListedColormap(custom_colors)
        
    ##############
    # Setup plot #
    ##############
    FONTSIZE = 4
    LINEWIDTH = 0.5
    plt.rcdefaults()
    sns.set_style("white", {"ytick.left": True, "xtick.top": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams["hatch.linewidth"] = 0.2
    plt.rcParams["axes.linewidth"] = LINEWIDTH
    
    # 2 x 2 as we draw the CPU heatmap in the top row, https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-and-subplots-py;
    fig, axes = plt.subplots(2, 3, sharex="col", figsize=(3.34, 1.6), dpi=200,
                             gridspec_kw={"width_ratios": [100, 100, 8], "height_ratios": [12.5, 100]})
    gs = axes[0, 2].get_gridspec()
    # Remove the existing axes in the right column;
    for ax in axes[0:, -1]:
        ax.remove()
    plt.subplots_adjust(top=0.86,
                        bottom=0.08,
                        left=0.05,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.0)
    # Create a large axis;
    ax_cbar = fig.add_subplot(gs[0:, 2])
    
    for i, (gpu, data) in enumerate(data_dict.items()):
        ax_gpu = axes[1, i]
        ax_cpu = axes[0, i]
        # GPU;
        ax_gpu, ax_cbar = plot_gpu(data, ax_gpu, i, ax_cbar if i == 0 else None)
        # CPU;
        ax_cpu = plot_cpu(data, ax_cpu, i)
        # GPU Label;
        ax_gpu.annotate(f"{gpu}", xy=(0.5, -0.1), fontsize=FONTSIZE + 2, ha="center", color="#2f2f2f", clip_on=False, xycoords="axes fraction")
    
    return fig

##############################
##############################

if __name__ == "__main__":
    
    d = {}
    for g in [V100, A100]:
    
        # Read data (TODO: use A100 data);
        data = pd.read_csv(os.path.join(DEFAULT_RES_DIR, DATASET.format(NUM_GPU, V100.lower())), names=["from", "to", "bandwidth"], skiprows=1)
        
        # Mock data for tha A100 (non-CPU interconnection is 2x faster);
        if g == A100:
            data.loc[data["from"] != -1, "bandwidth"] *= 2
        
        # Round to integer;
        data["bandwidth"] = data["bandwidth"].astype(int)
        for c in ["from", "to"]:
            # Replace "-1" with CPU and other numbers with the GPU name;
            data[c].replace({-1: "CPU", **{i: f"GPU{i}" for i in range(NUM_GPU)}}, inplace=True)
            # Use categorical labels for devices;
            data[c] = pd.Categorical(data[c], categories=["CPU"] + [f"GPU{i}" for i in range(NUM_GPU)], ordered=True)
        # Sort values;
        data.sort_values(["from", "to"], inplace=True)
        # Turn the dataframe into a matrix;
        data_matrix = data.pivot(index="from", columns="to", values="bandwidth")
        d[g] = data_matrix
        
    # Plot heatmap;
    fig = plot_heatmap(d)
    save_plot(PLOT_DIR, "bandwidth_gpus" + "_{}.{}", date=OUTPUT_DATE, dpi=600)

