# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:02:05 2022

@author: albyr
"""

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from load_data import (PLOT_DIR, BENCHMARK_NAMES, load_data_cuda_multigpu,
                       load_data_grcuda_multigpu)
import segretini_matplottini.src.plot_utils as pu
from plot_multi_gpu_cuda_exec_time import plot_speedup_bars, plot_speedup_line, plot_ablation_bars

##############################
##############################

OUTPUT_DATE = "2022_01_20"

# V100;
V100 = "V100"
V100_RES_FOLDERS_CUDA = [
    "2022_01_16_18_09_04_cuda_1-2gpu_v100",
    "2022_01_16_18_17_05_cuda_4gpu_v100",
    ]
V100_RES_FOLDERS_GRCUDA = [
    "2022_01_18_16_10_41_grcuda_1-2gpu_v100",
    "2022_01_18_10_01_23_grcuda_4gpu_v100",
    ]


class ResultGPU:
    def __init__(self, gpu: str, results_grcuda: list, results_cuda: list):
        self.gpu = gpu
        self.results_grcuda_folders = results_grcuda
        self.results_cuda_folders = results_cuda
        self.results_grcuda = None
        self.results_cuda = None
        self.results_merged = None
        
    @staticmethod
    def keep_useful_columns(data: pd.DataFrame, extra_columns: list=None, extra_index_columns: list=None) -> pd.DataFrame:
        useful_columns = ["benchmark", "size", "gpus", "exec_policy", "device_selection_policy",
                          "num_iter", "computation_sec", "speedup", "baseline_time"]
        index_columns = useful_columns[:6]
        if extra_columns is not None:
            useful_columns += [e for e in extra_columns if e not in useful_columns]
        if extra_index_columns is not None:
            index_columns += [e for e in extra_index_columns if (e not in index_columns) and (e in useful_columns)]
        # Drop any missing column;
        final_columns = [c for c in useful_columns if c in data.columns]
        data = data[final_columns]
        return data.sort_values(final_columns[:len(index_columns)]).reset_index(drop=True)
    
    def load_cuda(self):
        self.results_cuda = load_data_cuda_multigpu([os.path.join(self.gpu, f) for f in self.results_cuda_folders], skip_iter=3)
        return self.results_cuda
    
    def load_grcuda(self):
        self.results_grcuda = load_data_grcuda_multigpu([os.path.join(self.gpu, f) for f in self.results_grcuda_folders], skip_iter=3)
        return self.results_grcuda
    
    def group_grcuda_results(self, group_sizes: bool=False, drop_sync: bool=False, drop_nan: bool=True):
        if self.results_grcuda is None:
            self.load_grcuda()
        group_fields = ["benchmark", "exec_policy", "gpus"] + \
            (["size"] if not group_sizes else []) + \
            ["parent_stream_policy", "device_selection_policy"]
        grouped = self.results_grcuda.groupby(group_fields).mean()[["computation_sec", "speedup"]].reset_index()
        if drop_nan:
            grouped = grouped.dropna().reset_index(drop=True)
        if drop_sync:
            grouped = grouped[grouped["exec_policy"] != "SYNC"]
        return grouped 
    
    def join_grcuda_and_cuda_results(self):
        if self.results_grcuda is None:
            self.load_grcuda()
        if self.results_cuda is None:
            self.load_cuda()    
        res_merged = self.results_grcuda.merge(self.results_cuda, how="left",
                                               on=["benchmark", "size", "gpus", "exec_policy",
                                                   "prefetch", "num_blocks", "block_size_1d", 
                                                   "block_size_2d", "num_iter",
                                                   "total_iterations", "block_size_str"],
                                               suffixes=["_grcuda", "_cuda"])
        # Keep only the GrCUDA speedup vs. GrCUDA sync, and the raw execution time of GrCUDA and CUDA;
        res_merged.rename(columns={"speedup_grcuda": "speedup_grcuda_vs_grcuda_async", "speedup_cuda": "speedup_cuda_vs_cuda_async"}, inplace=True)
        columns_to_keep = [c for c in res_merged.columns if "cuda" not in c] + \
            ["computation_sec_grcuda", "computation_sec_cuda", "baseline_time_grcuda", "baseline_time_cuda"]
        res_merged = res_merged[columns_to_keep + ["speedup_grcuda_vs_grcuda_async", "speedup_cuda_vs_cuda_async"]]
        # Compute speedup of GrCUDA vs CUDA;
        res_merged["speedup_grcuda_vs_cuda"] = res_merged["computation_sec_cuda"] / res_merged["computation_sec_grcuda"]
        self.res_merged = res_merged
        return self.res_merged
    
    def group_merged_results(self, group_sizes: bool=False, drop_sync: bool=False, drop_nan: bool=True):
        if self.res_merged is None:
            self.join_grcuda_and_cuda_results()
        group_fields = ["benchmark", "exec_policy", "gpus"] + \
            (["size"] if not group_sizes else []) + \
            ["parent_stream_policy", "device_selection_policy"]
        grouped = self.res_merged.groupby(group_fields).mean()[["computation_sec_grcuda", "computation_sec_cuda", 
                                                                "speedup_grcuda_vs_grcuda_async", "speedup_cuda_vs_cuda_async", "speedup_grcuda_vs_cuda"]].reset_index()
        if drop_nan:
            grouped = grouped.dropna().reset_index(drop=True)
        if drop_sync:
            grouped = grouped[grouped["exec_policy"] != "SYNC"]
        return grouped 
    

V100_RESULTS = ResultGPU(
    gpu=V100,
    results_grcuda=V100_RES_FOLDERS_GRCUDA,
    results_cuda=V100_RES_FOLDERS_CUDA
    ) 

##############################
##############################   

def plot_grcuda_ablation(gpu_results: list[ResultGPU]):
    
    gpus = [2, 4]
    
    # Setup plot;
    plt.rcdefaults()
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": False})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams['axes.labelpad'] = 5
    plt.rcParams['xtick.major.pad'] = 4.2
    plt.rcParams['ytick.major.pad'] = 1
    plt.rcParams['axes.linewidth'] = 0.5
    
    fig = plt.figure(figsize=(3.4, 1.6), dpi=600)
    gs = gridspec.GridSpec(len(gpus), len(gpu_results))
    plt.subplots_adjust(top=0.85,
                        bottom=0.08,
                        left=0.05,
                        right=0.99,
                        hspace=0.2,
                        wspace=0.15)
    # Draw plot;
    for g_i, g in enumerate(gpu_results):
        data = g.join_grcuda_and_cuda_results()
        for g_j, num_gpus in enumerate(gpus):
            
            # Create new axis;
            ax = fig.add_subplot(gs[g_j, g_i])
            
            res_for_ablation = data.query(f"gpus == {num_gpus}")
            # FIXME: don't keep just these sizes;
            chosen_sizes = pd.DataFrame.from_dict({k: [v] for k, v in {"VEC": 950000000, "B&S": 35000000, "ML": 1200000, "CG": 50000, "MUL": 60000}.items()}, orient="index").reset_index().rename(columns={"index": "benchmark", 0: "size"})
            res_for_ablation = res_for_ablation.merge(chosen_sizes, how="inner", on=["benchmark", "size"])
            res_for_ablation["benchmark"] = pd.Categorical(res_for_ablation["benchmark"], list(BENCHMARK_NAMES.values()))
            # Clean data;
            res_for_ablation = ResultGPU.keep_useful_columns(res_for_ablation, extra_columns=["parent_stream_policy", "device_selection_policy", "computation_sec_grcuda"])
            res_for_ablation = res_for_ablation.groupby(["benchmark", "size", "gpus", "exec_policy", "parent_stream_policy", "device_selection_policy"]).mean().dropna().reset_index()
            # Compute speedup;
            pu.compute_speedup_df(res_for_ablation, ["benchmark"],
                                  baseline_filter_col=["parent_stream_policy", "device_selection_policy"],
                                  baseline_filter_val=["multigpu-disjoint", "minmax-transfer-time"],
                                  time_column="computation_sec_grcuda", aggregation=np.mean)
            # Add a new column to identify the policy, and make it categorical;
            res_for_ablation["policy"] = res_for_ablation["parent_stream_policy"] + "-" + res_for_ablation["device_selection_policy"]
            res_for_ablation["policy"] = pd.Categorical(res_for_ablation["policy"],
                                                        ["disjoint-round-robin", "disjoint-stream-aware", 
                                                         "disjoint-minmax-transfer-time", "multigpu-disjoint-minmax-transfer-time"])
            res_for_ablation = res_for_ablation.dropna().reset_index(drop=True)
            # Plot;
            plot_ablation_bars(res_for_ablation, g.gpu, num_gpus, fig=fig, ax=ax, ymax=1.2, yticks=7, plot_speedup_labels=False)
    pu.save_plot(PLOT_DIR, f"grcuda_ablation_{g.gpu}" + "_{}.{}", date=OUTPUT_DATE, dpi=600)

#%%###########################
##############################    

if __name__ == "__main__":
    g = V100_RESULTS
    res_cuda = g.load_cuda()
    res_grcuda = g.load_grcuda()
    res_grcuda_grouped = g.group_grcuda_results()
    res_grcuda_grouped_small = g.group_grcuda_results(group_sizes=True, drop_sync=True)
    res_merged = g.join_grcuda_and_cuda_results()
    res_merged_grouped = g.group_merged_results(group_sizes=True, drop_sync=True)
    
    #%% 1: Barplot with average speedup of GrCUDA vs async, 1 GPU. Consider only the best policy;
    res_for_barplot = res_merged[(res_merged["gpus"] == 1) | \
                                 ((res_merged["parent_stream_policy"] == "multigpu-disjoint") & \
                                 (res_merged["device_selection_policy"] == "minmax-transfer-time"))]
    # FIXME: CG and ML are incomplete. Keep only the specified sizes; 
    chosen_sizes = pd.DataFrame.from_dict({k: [v] for k, v in {"VEC": 950000000, "B&S": 35000000, "ML": 1200000, "CG": 50000, "MUL": 60000}.items()}, orient="index").reset_index().rename(columns={"index": "benchmark", 0: "size"})
    res_for_barplot = res_for_barplot.merge(chosen_sizes, how="inner", on=["benchmark", "size"])
    res_for_barplot["benchmark"] = pd.Categorical(res_for_barplot["benchmark"], list(BENCHMARK_NAMES.values()))
    # Simplify and aggregate;
    res_for_barplot = ResultGPU.keep_useful_columns(res_for_barplot, extra_columns=list(res_for_barplot.columns[-5:]))
    res_for_barplot = res_for_barplot.groupby(["benchmark", "size", "gpus", "exec_policy", "device_selection_policy"]).mean().dropna().reset_index()
    # Plot;
    plot_speedup_bars(res_for_barplot, f"GrCUDA, {g.gpu}", speedup_column="speedup_grcuda_vs_grcuda_async", baseline_is_async=True)
    pu.save_plot(PLOT_DIR, f"grcuda_vs_grcuda_bars_{g.gpu}" + "_{}.{}", date=OUTPUT_DATE, dpi=600)
    
    # Also plot results for CUDA;
    # FIXME: don't filter sizes;
    res_for_barplot_cuda = res_merged.merge(chosen_sizes, how="inner", on=["benchmark", "size"]) 
    res_for_barplot_cuda["benchmark"] = pd.Categorical(res_for_barplot_cuda["benchmark"], list(BENCHMARK_NAMES.values()))
    # Clean data;
    res_for_barplot_cuda = ResultGPU.keep_useful_columns(res_for_barplot_cuda, extra_columns=["speedup_cuda_vs_cuda_async", "baseline_time_cuda"])
    res_for_barplot_cuda = res_for_barplot_cuda.groupby(["benchmark", "size", "gpus", "exec_policy"]).mean().dropna().reset_index()
    # Plot;
    plot_speedup_bars(res_for_barplot_cuda, f"CUDA, {g.gpu}", speedup_column="speedup_cuda_vs_cuda_async", baseline_is_async=True)
    pu.save_plot(PLOT_DIR, f"cuda_vs_cuda_bars_{g.gpu}" + "_{}.{}", date=OUTPUT_DATE, dpi=600)
    
    #%% 2: Lineplot with speedup of GrCUDA, best policy, divided by size;
    res_for_lineplot = res_merged[(res_merged["gpus"] == 1) | \
                                 ((res_merged["parent_stream_policy"] == "multigpu-disjoint") & \
                                 (res_merged["device_selection_policy"] == "minmax-transfer-time"))]
    plot_speedup_line(res_for_lineplot, g.gpu, speedup_column="speedup_grcuda_vs_grcuda_async", baseline_time_column="baseline_time_grcuda", kind="GrCUDA")
    pu.save_plot(PLOT_DIR, f"grcuda_vs_grcuda_lines_{g.gpu}" + "_{}.{}", date=OUTPUT_DATE, dpi=600)
    plot_speedup_line(res_for_lineplot, g.gpu, speedup_column="speedup_cuda_vs_cuda_async", baseline_time_column="baseline_time_cuda")
    pu.save_plot(PLOT_DIR, f"cuda_vs_cuda_lines_{g.gpu}" + "_{}.{}", date=OUTPUT_DATE, dpi=600)
    
    #%% 3: Barplot of GrCUDA, best policy, versus CUDA;
    plot_speedup_bars(res_for_barplot, f"GrCUDA vs. CUDA, {g.gpu}", speedup_column="speedup_grcuda_vs_cuda", 
                      baseline_is_async=False, legend_title="Baseline: ASYNC, 1 GPU, CUDA",
                      legend_baseline_label="ASYNC, 1 GPU, GrCUDA", ymax=1.6, yticks=9)
    pu.save_plot(PLOT_DIR, f"grcuda_vs_cuda_bars_{g.gpu}" + "_{}.{}", date=OUTPUT_DATE, dpi=600)
    
    #%% 4: Barplot of GrCUDA, best policy, versus other policies. Do it just for 4 GPUs;
    plot_grcuda_ablation([g, g])
    
    