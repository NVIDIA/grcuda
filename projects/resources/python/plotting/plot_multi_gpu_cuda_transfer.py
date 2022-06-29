# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 12:54:09 2021

@author: albyr
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.gridspec as gridspec
from abc import ABC, abstractmethod
from segretini_matplottini.src.plot_utils import *
from multi_gpu_parse_nvprof_log import create_nondirectional_transfer_matrix
from load_data import PLOT_DIR, DEFAULT_RES_CUDA_DIR

##############################
# Setup ######################
##############################

# V100;
GPU = "V100"
INPUT_FOLDER = "V100/nvprof_cuda/2021_10_07"

# # A100;
# GPU = "A100"
# INPUT_FOLDER = "A100/nvprof_cuda/2021_10_18"

OUTPUT_DATE = "2021_10_18"

BENCHMARKS = [
    "b1",
    "b5",
    "b6",
    "b6_4",
    "b9",
    "b9_4",
    "b11"
    ]

##############################
# Drawers ####################
##############################

class GPUDrawer(ABC):
    
    @abstractmethod
    def setup(self, fig=None, ax=None):
        pass

    @abstractmethod
    def draw_topology(self, fig=None, ax=None, **kwargs):
        pass
    
    @abstractmethod
    def draw_transfer(self, ax, transfer_matrix_nondirectional, max_transfer: float=None, min_transfer: float=None, 
                      redraw_points: bool=True, **kwargs):
        pass

    @abstractmethod
    def add_benchmark_name(self, ax, b):
        pass
    
    @abstractmethod
    def setup_large_plot(self, num_benchmarks: int):
        pass


class V100Drawer(GPUDrawer):
    EDGE = 0.8
    ANGLE = 35
    FORSHORTENING = 1 / 3  # Proportional scaling of the axonometry;
    X_STEP = EDGE * FORSHORTENING
    Y_STEP = EDGE * FORSHORTENING * np.tan(np.deg2rad(ANGLE))
    CPU_VSTEP = EDGE / 3
     
    POINTS = [
        # Front face;
        [0, 0],  # 0: Lower left;
        [0, EDGE],  # 1: Upper left;
        [EDGE, EDGE],  # 2: Upper right;
        [EDGE, 0],  # 3: Lower right;
        # Right face;
        [EDGE + X_STEP, Y_STEP],  # 4: Lower
        [EDGE + X_STEP, EDGE + Y_STEP],  # 5: Upper
        # 6: Lower left corner;
        [X_STEP, Y_STEP],
        # 7: Upper left corner;
        [X_STEP, EDGE + Y_STEP], 
        ]
    
    # Associate GPUs to points;
    GPU = {
        0: POINTS[2],   
        1: POINTS[1],   
        2: POINTS[7],   
        3: POINTS[5],   
        4: POINTS[6],   
        5: POINTS[4],   
        6: POINTS[3],   
        7: POINTS[0],   
        }
    
    CPU_POINTS = [
        [EDGE / 2 + X_STEP / 2, Y_STEP * FORSHORTENING - CPU_VSTEP],
        [EDGE / 2 + X_STEP / 2, EDGE + Y_STEP * FORSHORTENING + CPU_VSTEP],
    ]
    
    CPU = {
       0: CPU_POINTS[0],    
       1: CPU_POINTS[1], 
       }       

    def setup(self, fig=None, ax=None):
        if fig == None and ax == None:
            plt.rcdefaults()
            plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
            # Create figure and axes
            fig, ax = plt.subplots()
        
        # Have axes with the same scale;
        plt.xlim(-0.6, self.EDGE * 3/2 + 0.4)
        plt.ylim(-0.6, self.EDGE * 3/2 + 0.4)
        ax.set_aspect("equal")
        plt.axis("off")
        return fig, ax
    
    # Draw corners of the cube;
    def draw_points(self, ax, alpha_scale=1):
        # Obtain list of coordinates;
        x = [x[0] for x in self.POINTS]
        y = [y[1] for y in self.POINTS]
        ax.scatter(x, y, color="#2f2f2f", alpha=alpha_scale, zorder=10)
        return ax
    
    # Draw names of GPUs;
    def draw_gpu_names(self, ax):
        x_offset = {
            2: -self.EDGE / 5,
            3: -self.EDGE / 10,
            6: -self.EDGE / 8,
            7: -self.EDGE / 5,
            }
        y_offset = {
            0: -self.EDGE / 10,
            1: -self.EDGE / 10,
            2: self.EDGE / 25,
            3: self.EDGE / 25,
            6: -self.EDGE / 8,
            7: -self.EDGE / 8,
            }
        for i, (g, p) in enumerate(self.GPU.items()):
            x = p[0] + self.EDGE * 0.02 + (x_offset[g] if g in x_offset else 0)
            y = p[1] + self.EDGE * 0.01 + (y_offset[g] if g in y_offset else self.EDGE / 70)
            ax.annotate(f"GPU{g}", xy=(x, y), color="#2f2f2f", fontsize=10, ha="left")
        return ax 
    
    # Draw a single line between GPUs;
    def draw_line_gpu(self, ax, x, y, style):
            x = self.GPU[x]
            y = self.GPU[y]
            ax.plot((x[0], y[0]), (x[1], y[1]), **style)
                 
    # Join corners;
    def draw_edges(self, ax, alpha_scale=1):    
        # Double NVLink;
        style_nv2 = dict(
            linewidth=2,
            linestyle="-",
            color="#2f2f2f",
            alpha=0.9 * alpha_scale,
            solid_capstyle="round",
        )
        # Single NVLink;
        style_nv1 = dict(
            linewidth=0.8,
            linestyle="--",
            color="#2f2f2f",
            alpha=0.7 * alpha_scale,
            solid_capstyle="round",
        )
        # Missing edge is PCIe;  
        
        # Connect GPUs;
        self.draw_line_gpu(ax, 0, 1, style_nv1)
        self.draw_line_gpu(ax, 0, 2, style_nv1)
        self.draw_line_gpu(ax, 0, 3, style_nv2)
        self.draw_line_gpu(ax, 0, 6, style_nv2)
        self.draw_line_gpu(ax, 1, 2, style_nv2)
        self.draw_line_gpu(ax, 1, 3, style_nv1)
        self.draw_line_gpu(ax, 1, 7, style_nv2)
        self.draw_line_gpu(ax, 2, 3, style_nv2)
        self.draw_line_gpu(ax, 2, 4, style_nv1)
        self.draw_line_gpu(ax, 3, 5, style_nv1)
        self.draw_line_gpu(ax, 4, 5, style_nv2)
        self.draw_line_gpu(ax, 4, 6, style_nv1)
        self.draw_line_gpu(ax, 4, 7, style_nv2)
        self.draw_line_gpu(ax, 5, 6, style_nv2)
        self.draw_line_gpu(ax, 5, 7, style_nv1)
        self.draw_line_gpu(ax, 6, 7, style_nv1)
        
        return ax
    
    # Draw faces of the cube;
    def draw_faces(self, ax):
        style = dict(
            linewidth=1,
            linestyle="--",
            edgecolor="#2f2f2f",
            facecolor="#2f2f2f",
            alpha=0.1,
        )
        points = self.POINTS
        patches_list = [
            patches.Polygon(xy=[points[0], points[1], points[2], points[3]], **style),
            patches.Polygon(xy=[points[2], points[5], points[4], points[3]], **style),
            patches.Polygon(xy=[points[2], points[5], points[7], points[1]], **style),
            patches.Polygon(xy=[points[0], points[3], points[4], points[6]], **style),
            patches.Polygon(xy=[points[0], points[1], points[7], points[6]], **style),
            patches.Polygon(xy=[points[6], points[4], points[5], points[7]], **style),
            ]
        for p in patches_list:
            ax.add_patch(p)
        return ax
    
    def draw_cpu_points(self, ax, alpha_scale=1): 
        # Obtain list of coordinates;
        x = [x[0] for x in self.CPU_POINTS]
        y = [y[1] for y in self.CPU_POINTS]
        ax.scatter(x, y, color="#888888", alpha=alpha_scale, zorder=10)
        return ax
    
    def draw_pci(self, ax, gpu0, gpu1, vertical_start, upper=True, style=None):
        medium_point = [(gpu0[0] + gpu1[0]) / 2, gpu0[1]]
        x_step = self.X_STEP / 2 - gpu0[0]
        y_step = self.Y_STEP * self.FORSHORTENING + self.CPU_VSTEP * (1 if upper else -1)
        cpu_point = [medium_point[0] + x_step, vertical_start + y_step]    
        
        t = np.sqrt(y_step**2 + x_step**2) / 2
        alpha = np.arctan(y_step / np.abs(x_step))
        y_offset = np.sin(alpha) * t
        
        x_offset = (cpu_point[0] - medium_point[0]) / 2
        split_point = [medium_point[0] + x_offset, vertical_start + y_offset + (gpu0[1] - vertical_start) / 2]
        
        ax.plot((cpu_point[0], split_point[0]), (cpu_point[1], split_point[1]), **style)
        ax.plot((split_point[0], gpu0[0]), (split_point[1], gpu0[1]), **style)
        ax.plot((split_point[0], gpu1[0]), (split_point[1], gpu1[1]), **style)        
    
    def draw_cpu_lines(self, ax, alpha_scale=1):
        style = dict(
            color="#888888",
            alpha=0.8 * alpha_scale,
            linestyle="-",
            linewidth=1,
            solid_capstyle="round",
        )
                    
        self.draw_pci(ax, self.GPU[1], self.GPU[0], self.EDGE, style=style)
        self.draw_pci(ax, self.GPU[2], self.GPU[3], self.EDGE, style=style)
        self.draw_pci(ax, self.GPU[7], self.GPU[6], 0, False, style=style)
        self.draw_pci(ax, self.GPU[4], self.GPU[5], 0, False, style=style)
        return ax
    
    # Draw names of CPUs;
    def draw_cpu_names(self, ax):
        y_offset = {
            0: -self.EDGE / 10,
            }
        for c in [0, 1][::-1]:
            p = self.CPU_POINTS[c]
            x = p[0] + self.EDGE * 0.02
            y = p[1] + self.EDGE * 0.01 + (y_offset[c] if c in y_offset else self.EDGE / 70)
            ax.annotate(f"CPU{c}", xy=(x, y), color="#888888", fontsize=10, ha="left")
        return ax
    
    # Draw the GPU topology;
    def draw_topology(self, fig=None, ax=None, **kwargs):
        fig, ax = self.setup(fig, ax)
        ax = self.draw_cpu_lines(ax, **kwargs)
        ax = self.draw_edges(ax, **kwargs)
        # ax = draw_faces(ax)
        ax = self.draw_points(ax, **kwargs)
        ax = self.draw_cpu_points(ax, **kwargs)
        ax = self.draw_gpu_names(ax)
        ax = self.draw_cpu_names(ax)
        return fig, ax
    
    def draw_pci_transfer(self, ax, cpu, gpu, other_gpu, vertical_start, upper=True, style=None):
        medium_point = [(gpu[0] + other_gpu[0]) / 2, gpu[1]]
        x_step = self.X_STEP / 2 - min(gpu[0], other_gpu[0])
        y_step = self.Y_STEP * self.FORSHORTENING + self.CPU_VSTEP * (1 if upper else -1)
        cpu_point = [medium_point[0] + x_step, vertical_start + y_step]    
        
        t = np.sqrt(y_step**2 + x_step**2) / 2
        alpha = np.arctan(y_step / np.abs(x_step))
        y_offset = np.sin(alpha) * t
        
        x_offset = (cpu_point[0] - medium_point[0]) / 2
        split_point = [medium_point[0] + x_offset, vertical_start + y_offset + (gpu[1] - vertical_start) / 2]
        ax.plot((split_point[0], gpu[0]), (split_point[1], gpu[1]), **style)
    
    def draw_pci_transfer_cpu(self, ax, cpu, gpu, other_gpu, vertical_start, upper=True, style=None, zorder=None):
        medium_point = [(gpu[0] + other_gpu[0]) / 2, gpu[1]]
        x_step = self.X_STEP / 2 - min(gpu[0], other_gpu[0])
        y_step = self.Y_STEP * self.FORSHORTENING + self.CPU_VSTEP * (1 if upper else -1)
        cpu_point = [medium_point[0] + x_step, vertical_start + y_step]    
        
        t = np.sqrt(y_step**2 + x_step**2) / 2
        alpha = np.arctan(y_step / np.abs(x_step))
        y_offset = np.sin(alpha) * t
        
        x_offset = (cpu_point[0] - medium_point[0]) / 2
        split_point = [medium_point[0] + x_offset, vertical_start + y_offset + (gpu[1] - vertical_start) / 2]
        ax.plot((cpu_point[0], split_point[0]), (cpu_point[1], split_point[1]), zorder=zorder, **style)
    
    # Draw the transfer between devices;
    def draw_transfer(self, ax, transfer_matrix_nondirectional, max_transfer: float=None, min_transfer: float=None, 
                      redraw_points: bool=True, **kwargs):
       
        PALETTE = sns.color_palette("YlOrBr", as_cmap=True)# sns.color_palette("YlOrBr", as_cmap=True)
        MIN_PAL = 0.2
        MAX_PAL = 0.5
        MAX_WIDTH = 4
        MIN_WIDTH = 0.5
        if max_transfer is None:
            max_transfer = transfer_matrix_nondirectional.max().max()
        if min_transfer is None:
            min_transfer = transfer_matrix_nondirectional.min().min()
            
        def style_gpu(transfer):
            return dict(
                linewidth=transfer * (MAX_WIDTH - MIN_WIDTH) + MIN_WIDTH,
                linestyle="-",
                color=PALETTE(transfer * (MAX_PAL - MIN_PAL) + MIN_PAL),
                alpha=0.7,
                solid_capstyle="round",
            )
        
        # Shared PCI express channels;
        total_pci_01 = transfer_matrix_nondirectional.loc[transfer_matrix_nondirectional.index.isin(["0", "1"])]["CPU"].sum()
        total_pci_23 = transfer_matrix_nondirectional.loc[transfer_matrix_nondirectional.index.isin(["2", "3"])]["CPU"].sum()   
        total_pci_45 = transfer_matrix_nondirectional.loc[transfer_matrix_nondirectional.index.isin(["4", "5"])]["CPU"].sum()
        total_pci_67 = transfer_matrix_nondirectional.loc[transfer_matrix_nondirectional.index.isin(["6", "7"])]["CPU"].sum()
        self.draw_pci_transfer_cpu(ax, self.CPU[0], self.GPU[1], self.GPU[0], self.EDGE, style=style_gpu(total_pci_01), zorder=9)
        self.draw_pci_transfer_cpu(ax, self.CPU[0], self.GPU[3], self.GPU[2], self.EDGE, style=style_gpu(total_pci_23), zorder=9)
        self.draw_pci_transfer_cpu(ax, self.CPU[1], self.GPU[4], self.GPU[5], 0, False, style=style_gpu(total_pci_45))
        self.draw_pci_transfer_cpu(ax, self.CPU[1], self.GPU[7], self.GPU[6], 0, False, style=style_gpu(total_pci_67))
        
        # All the other channels;  
        for ii, i in enumerate(transfer_matrix_nondirectional.index):
            for jj, j in enumerate(transfer_matrix_nondirectional.columns):
                # Symmetric matrix, the lower triangular part is skipped;
                if ii > jj:
                    continue
                transfer = transfer_matrix_nondirectional.loc[i, j]
                if transfer > 0:
                    # Draw GPU-GPU transfer;
                    if i != "CPU" and j != "CPU":
                        self.draw_line_gpu(ax, int(i), int(j), style_gpu(transfer))
                    # Draw CPU-GPU transfer;
                    else:
                        if j == "CPU":
                            gpu = int(i)
                            if gpu < 4:
                                self.draw_pci_transfer(ax, self.CPU[0], self.GPU[gpu], self.GPU[(gpu + 1) % 2 + (2 if gpu > 1 else 0)], self.EDGE, style=style_gpu(transfer))
                            elif gpu >= 4:
                                self.draw_pci_transfer(ax, self.CPU[1], self.GPU[gpu], self.GPU[(gpu + 1) % 2 + (6 if gpu > 5 else 4)], 0, False, style=style_gpu(transfer))
        return ax
   
    def add_benchmark_name(self, ax, b):
        ax.annotate(b.upper(), xy=(0.78, 0.85), xycoords="axes fraction", ha="left", color="#2f2f2f", fontsize=14, alpha=1)   
        return ax
    
    def setup_large_plot(self, num_benchmarks: int):
        plt.rcdefaults()
        plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
        
        cols = num_benchmarks
        rows = 1
        scale = 2.5
        fig, ax = plt.subplots(figsize=(cols * scale, scale))
        gs = gridspec.GridSpec(rows, cols)
        plt.subplots_adjust(
                    top=1,
                    bottom=0,
                    left=0,
                    right=0.992,
                    hspace=0.01,
                    wspace=0.01)
        plt.axis("off")
        return fig, ax, gs
    
    
class A100Drawer(GPUDrawer):
    
    NUM_GPUS = 8
    NUM_CPUS = 2
    X_MIN = 0
    X_MAX = 3.8
    Y_MIN = 0
    Y_MAX = 1.2
    
    X_RANGE = X_MAX - X_MIN
    Y_RANGE = Y_MAX - Y_MIN
    X_OFFSET = X_RANGE * 0.05
    GPU_RANGE = (X_RANGE - 2 * X_OFFSET)
    X_SHIFT = GPU_RANGE / NUM_GPUS
    Y_GPU = 0.5 * Y_RANGE
    Y_CPU = 0.9 * Y_RANGE
    GPU_GROUP_SIZE = NUM_GPUS // NUM_CPUS
    STEP_SIZE_GPU = GPU_RANGE / (NUM_GPUS - 1)
    X_OFFSET_CPU = X_OFFSET + STEP_SIZE_GPU * (GPU_GROUP_SIZE - 1) / 2
    Y_NVSWITCH = 0.1 * Y_RANGE
    Y_OFFSET_NVSWITCH = Y_GPU - Y_NVSWITCH
    
    X_GPU_POINTS = np.linspace(X_OFFSET, X_RANGE - X_OFFSET, NUM_GPUS)
    Y_GPU_POINTS = [Y_GPU] * NUM_GPUS
    X_CPU_POINTS = np.linspace(X_OFFSET_CPU, X_RANGE - X_OFFSET_CPU, NUM_CPUS)
    Y_CPU_POINTS = [Y_CPU] * NUM_CPUS
    CPU = [[x, y] for x, y in zip(X_CPU_POINTS, Y_CPU_POINTS)]
    GPU = [[x, y] for x, y in zip(X_GPU_POINTS, Y_GPU_POINTS)]
    
    def setup(self, fig=None, ax=None):
        if fig == None and ax == None:
            plt.rcdefaults()
            plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(self.X_RANGE, self.Y_RANGE))
            plt.subplots_adjust(
                        top=1,
                        bottom=0,
                        left=0,
                        right=0.94)
            
        # Have axes with the same scale;
        plt.xlim(self.X_MIN, self.X_MAX)
        plt.ylim(self.Y_MIN, self.Y_MAX)
        ax.set_aspect("equal")
        plt.axis("off")
        return fig, ax
    
    def draw_points(self, ax, alpha_scale=1):
        ax.scatter(self.X_GPU_POINTS, self.Y_GPU_POINTS, color="#2f2f2f", alpha=alpha_scale, zorder=100)                
        return ax
    
    def draw_cpu_points(self, ax, alpha_scale=1): 
        ax.scatter(self.X_CPU_POINTS, self.Y_CPU_POINTS, edgecolor="#888888", color="w", alpha=alpha_scale, zorder=100)         
        return ax
    
    def draw_cpu_lines(self, ax, alpha_scale=1):
        style = dict(
            color="#888888",
            alpha=0.8 * alpha_scale,
            linestyle="-",
            linewidth=1,
            solid_capstyle="round",
        )
        style_cpu_link = dict(
            color="#888888",
            alpha=0.8 * alpha_scale,
            linestyle=":",
            linewidth=1,
            solid_capstyle="round",
        )
        
        for cpu in range(self.NUM_CPUS):
            for gpu in range(self.GPU_GROUP_SIZE):
                gpu_tot = gpu + cpu * self.GPU_GROUP_SIZE
                ax.plot((self.CPU[cpu][0], self.GPU[gpu_tot][0]), (self.CPU[cpu][1], self.GPU[gpu_tot][1]), **style)
        ax.plot((self.CPU[0][0], self.CPU[-1][0]), (self.CPU[-1][1], self.CPU[-1][1]), **style_cpu_link)        
        
        return ax
    
    def draw_gpu_lines(self, ax, alpha_scale=1):
        style_nv = dict(
            linewidth=1,
            linestyle="-",
            color="#2f2f2f",
            alpha=1 * alpha_scale,
            solid_capstyle="round",
        )
        style_switch = dict(
            linewidth=5,
            linestyle="-",
            color="#2f2f2f",
            alpha=1 * alpha_scale,
            solid_capstyle="round",
        )
        
        for gpu in range(self.NUM_GPUS):
            ax.plot((self.GPU[gpu][0], self.GPU[gpu][0]), (self.Y_NVSWITCH, self.GPU[gpu][1]), **style_nv)
        ax.plot((self.X_OFFSET, self.X_OFFSET + self.GPU_RANGE), (self.Y_NVSWITCH, self.Y_NVSWITCH), **style_switch)
        return ax
    
    # Draw names of CPUs;
    def draw_cpu_names(self, ax):
        for i, c in enumerate(self.CPU):
            x = c[0] + self.X_RANGE * 0.01
            y = c[1] + self.Y_RANGE * 0.01
            ax.annotate(f"CPU{i}", xy=(x, y), color="#2f2f2f", fontsize=9, ha="left")
        return ax
    
    # Draw names of GPUs;
    def draw_gpu_names(self, ax):
        for i, g in enumerate(self.GPU):
            x = g[0] + self.X_RANGE * 0.005
            y = g[1] - self.Y_RANGE * 0.14
            ax.annotate(f"GPU{i}", xy=(x, y), color="#2f2f2f", fontsize=9, ha="left")
        return ax 
    
    def draw_other_names(self, ax):
        x = self.GPU[-1][0] + self.X_RANGE * 0.005
        y = self.Y_NVSWITCH + self.Y_RANGE * 0.04
        ax.annotate("NV12", xy=(x, y), color="#888888", fontsize=7, ha="left")
        
        x = self.GPU[0][0] + self.X_RANGE * 0.005
        y = self.Y_NVSWITCH - self.Y_RANGE * 0.1
        ax.annotate("NVSwitch", xy=(x, y), color="#888888", fontsize=7, ha="left")
        
        x = np.mean([c[0] for c in self.CPU]) 
        y = self.CPU[0][1] - self.Y_RANGE * 0.08
        ax.annotate("Infinity Fabric", xy=(x, y), ha="center", color="#888888", fontsize=7)
        
        x = np.mean([self.CPU[0][0], self.GPU[0][0]]) - self.X_RANGE * 0.02
        y = np.mean([self.CPU[0][1], self.GPU[0][1]]) + self.Y_RANGE * 0.02
        angle = np.rad2deg(np.arctan((self.CPU[0][1] - self.GPU[0][1]) / (self.CPU[0][0] - self.GPU[0][0])))
        ax.annotate("PCIe 4.0 x16", xy=(x, y), ha="center", color="#888888",
                    fontsize=7, rotation=angle, rotation_mode="anchor")
        
        return ax 

    def draw_topology(self, fig=None, ax=None, **kwargs):
        fig, ax = self.setup(fig, ax)
        ax = self.draw_cpu_lines(ax, **kwargs)
        ax = self.draw_gpu_lines(ax, **kwargs)
        ax = self.draw_points(ax, **kwargs)
        ax = self.draw_cpu_points(ax, **kwargs)
        ax = self.draw_gpu_names(ax)
        ax = self.draw_cpu_names(ax)
        ax = self.draw_other_names(ax)
        return fig, ax
    
    def draw_transfer(self, ax, transfer_matrix_nondirectional, max_transfer: float=None, min_transfer: float=None, 
                      redraw_points: bool=True, **kwargs):
        PALETTE = sns.color_palette("YlOrBr", as_cmap=True)
        MIN_PAL = 0.2
        MAX_PAL = 0.5
        MAX_WIDTH = 6
        MIN_WIDTH = 1.5
        if max_transfer is None:
            max_transfer = transfer_matrix_nondirectional.max().max()
        if min_transfer is None:
            min_transfer = transfer_matrix_nondirectional.min().min()
            
        def style_gpu(transfer):
            return dict(
                linewidth=transfer * (MAX_WIDTH - MIN_WIDTH) + MIN_WIDTH,
                linestyle="-",
                color=PALETTE(transfer * (MAX_PAL - MIN_PAL) + MIN_PAL),
                alpha=0.7,
                solid_capstyle="round",
            )
        
        # PCI express channels;
        for cpu in range(self.NUM_CPUS):
            for gpu in range(self.GPU_GROUP_SIZE):
                gpu_tot = gpu + cpu * self.GPU_GROUP_SIZE
                if str(gpu_tot) in transfer_matrix_nondirectional.index:
                    cpu_gpu_transfer = transfer_matrix_nondirectional.loc[str(gpu_tot), :]["CPU"]
                    if cpu_gpu_transfer > 0:
                        ax.plot((self.CPU[cpu][0], self.GPU[gpu_tot][0]), (self.CPU[cpu][1], self.GPU[gpu_tot][1]), **style_gpu(cpu_gpu_transfer))
        
        # All the other channels;
        for gpu in range(self.NUM_GPUS):
            if str(gpu) in transfer_matrix_nondirectional.index and "NVSwitch" in transfer_matrix_nondirectional.index:
                gpu_switch_transfer = transfer_matrix_nondirectional.loc[str(gpu), :]["NVSwitch"]
                if gpu_switch_transfer > 0:
                    ax.plot((self.GPU[gpu][0], self.GPU[gpu][0]), (self.Y_NVSWITCH, self.GPU[gpu][1]), **style_gpu(gpu_switch_transfer))
        if "NVSwitch" in transfer_matrix_nondirectional.index:
            switch_transfer_tot = transfer_matrix_nondirectional["NVSwitch"].sum()
            if switch_transfer_tot > 0:
                ax.plot((self.X_OFFSET, self.X_OFFSET + self.GPU_RANGE), (self.Y_NVSWITCH, self.Y_NVSWITCH), **style_gpu(switch_transfer_tot))
        return ax

    def add_benchmark_name(self, ax, b):
        ax.annotate(b.upper(), xy=(0.9, 0.88), xycoords="axes fraction", ha="left", color="#2f2f2f", fontsize=14, alpha=1)   
        return ax
      
    def setup_large_plot(self, num_benchmarks: int):
        plt.rcdefaults()
        plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
        
        cols = num_benchmarks
        rows = 1
        
        fig, ax = plt.subplots(figsize=(self.X_RANGE * cols, self.Y_RANGE))
        gs = gridspec.GridSpec(rows, cols)
        plt.subplots_adjust(
                    top=1,
                    bottom=0,
                    left=0,
                    right=0.992)
        plt.axis("off")
        return fig, ax, gs
    


##############################
##############################


if __name__ == "__main__":
    
    if GPU == "V100":
        gpu_drawer = V100Drawer()
    elif GPU == "A100":
        gpu_drawer = A100Drawer()
    else:
        raise ValueError(f"Unknown GPU={GPU}")
    
    fig, ax = gpu_drawer.draw_topology(alpha_scale=1)
    save_plot(PLOT_DIR, f"{GPU.lower()}_topology" + "_{}.{}", date=OUTPUT_DATE, dpi=600)    
    
    #%% Draw transfer of GPUs;
    
    # Obtain transfer max and min to normalize plots;
    maximum_transfer = 0
    minimum_transfer = np.inf
    num_benchmarks = 0
    for b in BENCHMARKS:
        transfer_matrix = pd.read_csv(os.path.join(DEFAULT_RES_CUDA_DIR, INPUT_FOLDER, b + "_transfer_matrix.csv"), index_col=0)
        maximum_transfer = max(transfer_matrix.max().max(), maximum_transfer)
        minimum_transfer = min(transfer_matrix.min().min(), minimum_transfer)
        num_benchmarks += 1
    
    #%%  A plot for every benchmark;
    for b in BENCHMARKS:
        fig, ax = gpu_drawer.draw_topology(alpha_scale=1)
        transfer_matrix = pd.read_csv(os.path.join(DEFAULT_RES_CUDA_DIR, INPUT_FOLDER, b + "_transfer_matrix.csv"), index_col=0)
        # Create non-directional matrix;
        transfer_matrix_nondirectional = create_nondirectional_transfer_matrix(transfer_matrix)
        # Normalize matrix;
        transfer_matrix_nondirectional /= transfer_matrix_nondirectional.max().max()
        # Draw colored edges;
        ax = gpu_drawer.draw_transfer(ax, transfer_matrix_nondirectional, max_transfer=maximum_transfer, min_transfer=minimum_transfer)
        # Add benchmark name;
        ax = gpu_drawer.add_benchmark_name(ax, b)
        save_plot(PLOT_DIR, f"{GPU.lower()}_topology_{b}" + "_{}.{}", date=OUTPUT_DATE, dpi=600)    
        
    #%% A single plot with all benchmarks;
    
    fig, ax, gs = gpu_drawer.setup_large_plot(num_benchmarks)
    
    for bi, b in enumerate(BENCHMARKS):
        ax = fig.add_subplot(gs[0, bi])
        fig, ax = gpu_drawer.draw_topology(alpha_scale=1, fig=fig, ax=ax)
        transfer_matrix = pd.read_csv(os.path.join(DEFAULT_RES_CUDA_DIR, INPUT_FOLDER, b + "_transfer_matrix.csv"), index_col=0)
        # Create non-directional matrix;
        transfer_matrix_nondirectional = create_nondirectional_transfer_matrix(transfer_matrix)
        # Normalize matrix;
        transfer_matrix_nondirectional /= transfer_matrix_nondirectional.max().max()
        # Draw colored edges;
        ax = gpu_drawer.draw_transfer(ax, transfer_matrix_nondirectional, max_transfer=maximum_transfer, min_transfer=minimum_transfer)
        # Add benchmark name;
        ax = gpu_drawer.add_benchmark_name(ax, b)
        save_plot(PLOT_DIR, f"{GPU.lower()}_topology" + "_{}.{}", date=OUTPUT_DATE, dpi=600) 
        
    