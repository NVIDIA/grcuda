#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:43:46 2020

@author: alberto.parravicini
"""

import pandas as pd
import json
import os
import numpy as np
import functools
from scipy.stats.mstats import gmean

DEFAULT_RES_DIR = "../../../../grcuda-data/results/scheduling"

# ASYNC_POLICY_NAME = "async"   # If parsing new results;
ASYNC_POLICY_NAME = "default"  # If parsing older results;

def load_data(input_date: str, skip_iter=0, remove_inf=True, remove_time_zero=True, benchmark="", phases=None) -> pd.DataFrame:
    """
    Load the benchmark results located in the input sub-folder
    :param input_date: name of the folder where results are located, as a subfolder of DEFAULT_RES_DIR
    :param skip_iter: skip the first iterations for each benchmark, as they are considered warmup
    :param remove_inf: remove rows with infinite speedup value, as they are not useful
    :param remove_time_zero: if True, remove rows with 0 computation time;
    :param benchmark: load data only for the specified benchmark
    :param phases: list of benchmark phases to add as columns
    :return: a DataFrame containing the results
    """
    input_path = os.path.join(DEFAULT_RES_DIR, input_date)

    # Load results as JSON;
    data_dict = {}
    for res in os.listdir(input_path):
        with open(os.path.join(input_path, res)) as f:
            if not benchmark or res.split("_")[6] == benchmark:
                data_dict[res] = json.load(f)
                
    phases_names = []

    # Turn results into a pd.DataFrame;
    rows = []
    for k, v in data_dict.items():
        row = []
        # Parse filename;
        benchmark, exec_policy, new_stream_policy, parent_stream_policy, dependency_policy, force_prefetch = k.split("_")[6:12]
        
        force_prefetch = force_prefetch == "True"
       
        row += [benchmark, exec_policy, new_stream_policy, parent_stream_policy, dependency_policy, force_prefetch, 0, 0, ""]

        # Retrieve other information;
        total_iterations = v["num_iterations"]
        cpu_validation = v["cpu_validation"]
        random_init = v["random_init"]
        size_dict = v["benchmarks"][benchmark][ASYNC_POLICY_NAME]
        row += [int(total_iterations), bool(cpu_validation), bool(random_init)]

        # Parse data for each input data size, and other settings;;
        for size, val_size in size_dict.items():
            for realloc, val_realloc in val_size.items():
                for reinit, val_reinit in val_realloc.items():
                    for block_size, val_block_size in val_reinit.items():
                        # Process each iteration;
                        block_size_1d, block_size_2d = block_size.split(",")
                        row[-6] = int(block_size_1d)
                        row[-5] = int(block_size_2d)
                        row[-4] = block_size_1d + ",8" # block_size_1d + "," + block_size_2d]
                        for curr_iteration in val_block_size:
                            num_iter = curr_iteration["iteration"]
                            gpu_result = curr_iteration["gpu_result"]
                            total_time_sec = curr_iteration["total_time_sec"]
                            overhead_sec = curr_iteration["overhead_sec"]
                            computation_sec = curr_iteration["computation_sec"]
                            
                            # Process phases;
                            phases_time = []
                            if phases:
                                phases_time = [p["time_sec"] for p in curr_iteration["phases"] if p["name"] in phases]
                                if not phases_names:
                                    phases_names = [p["name"] for p in curr_iteration["phases"] if p["name"] in phases]
                            
                            # Add a new row;
                            if (num_iter >= skip_iter):
                                rows += [row + [int(size), bool(realloc), bool(reinit), num_iter - skip_iter, gpu_result, total_time_sec, overhead_sec, computation_sec] + phases_time]

    columns = ["benchmark", "exec_policy", "new_stream_policy", "parent_stream_policy",
               "dependency_policy", "force_prefetch", "block_size_1d", "block_size_2d", "block_size_str",
               "total_iterations", "cpu_validation", "random_init", "size", "realloc", "reinit",
               "num_iter", "gpu_result", "total_time_sec", "overhead_sec", "computation_sec"] + (phases_names if phases else [])
    data = pd.DataFrame(rows, columns=columns).sort_values(by=columns[:14], ignore_index=True)
    
    # Clean columns with 0 computation time;
    if remove_time_zero:
        data = data[data["computation_sec"] > 0].reset_index(drop=True)
    
    # Compute speedups;
    compute_speedup(data, ["benchmark", "new_stream_policy", "parent_stream_policy",
               "dependency_policy", "force_prefetch", "block_size_1d", "block_size_2d",
               "total_iterations", "cpu_validation", "random_init", "size", "realloc", "reinit"])
    # Clean columns with infinite speedup;
    if remove_inf:
        data = data[data["computation_speedup"] != np.inf].reset_index(drop=True)
    
    return data


def load_data_cuda(input_date: str, skip_iter=0, remove_inf=True, remove_time_zero=True, add_prefetch_as_policy=True) -> pd.DataFrame:
    """
    Load the benchmark results located in the input sub-folder
    :param input_date: name of the folder where results are located, as a subfolder of DEFAULT_RES_DIR
    :param skip_iter: skip the first iterations for each benchmark, as they are considered warmup
    :param remove_inf: if True, remove rows with infinite speedup
    :param remove_time_zero: if True, remove rows with 0 computation time;
    :param add_prefetch_as_policy: if True, consider prefetching as part of the policy, to compute speedups w.r.t. sync with no prefetching
    :return: a DataFrame containing the results
    """
    input_path = os.path.join(DEFAULT_RES_DIR, input_date)

    # Load results as pd.DataFrames;
    data_tmp = []
    for f in os.listdir(input_path):
        # Parse filename;
        try:
            benchmark, exec_policy, size, block_size_1d, block_size_2d, force_prefetch, total_iterations, num_blocks = os.path.splitext(f)[0].split("_")[7:]
            force_prefetch = force_prefetch == "True"
        except ValueError:
            benchmark, exec_policy, size, block_size_1d, block_size_2d, total_iterations, num_blocks, force_prefetch = os.path.splitext(f)[0].split("_")[7:] + [False]
        tmp_data = pd.read_csv(os.path.join(input_path, f))
        
        # Skip first lines;
        tmp_data = tmp_data.iloc[skip_iter:, :]

        # Add other information;
        tmp_data["benchmark"] = benchmark
        tmp_data["exec_policy"] = exec_policy
        tmp_data["force_prefetch"] = bool(force_prefetch)
        tmp_data["size"] = int(size)
        tmp_data["block_size_1d"] = int(block_size_1d)
        tmp_data["block_size_2d"] = int(block_size_2d)
        tmp_data["block_size_str"] = block_size_1d + ",8" # block_size_1d + "," + block_size_2d
        tmp_data["total_iterations"] = int(total_iterations)
        data_tmp += [tmp_data]
        
    data = pd.concat(data_tmp).reset_index(drop=True)
    data["num_iter"] -= skip_iter

    # Reorder columns;
    columns = ["benchmark", "exec_policy", "force_prefetch", "block_size_1d", "block_size_2d", "block_size_str",
               "total_iterations", "size", "num_iter", "gpu_result", "total_time_sec", "overhead_sec", "computation_sec"]
    data = data[columns]
    
    # Clean columns with 0 computation time;
    if remove_time_zero:
        data = data[data["computation_sec"] > 0].reset_index(drop=True)
   
    # Compute speedups;
    if add_prefetch_as_policy:
        data["exec_policy_full"] = data["exec_policy"] + np.where(data["force_prefetch"], "_f", "")
        compute_speedup(data, ["benchmark", "block_size_1d", "block_size_2d", "size"], baseline_filter_col="exec_policy_full", baseline_filter_val="sync")
    else:
        compute_speedup(data, ["benchmark", "force_prefetch", "block_size_1d", "block_size_2d", "size"])
    
    # Clean columns with infinite speedup;
    if remove_inf:
        data = data[data["computation_speedup"] != np.inf].reset_index(drop=True)
    
    return data


def compute_speedup(data, key, speedup_col_name="computation_speedup", time_column="computation_sec",
                    baseline_filter_col="exec_policy", baseline_filter_val="sync", baseline_col_name="baseline_time_sec",
                    correction=True, aggregation=np.median):
    
    # Initialize speedup values;
    data[speedup_col_name] = 1
    data[baseline_col_name] = 0
    
    updated_groups = []
    grouped_data = data.groupby(key, as_index=False)
    for group_key, group in grouped_data:
        # Compute the median baseline computation time;
        median_baseline = aggregation(group.loc[group[baseline_filter_col] == baseline_filter_val, time_column])
        # Compute the speedup for this group;
        group.loc[:, speedup_col_name] = median_baseline / group[time_column]
        group.loc[:, baseline_col_name] = median_baseline
        data.loc[group.index, :] = group
    
        # Guarantee that the geometric mean of speedup referred to the baseline is 1, and adjust speedups accordingly;
        if correction:
            gmean_speedup = gmean(group.loc[group[baseline_filter_col] == baseline_filter_val, speedup_col_name])
            group.loc[:, speedup_col_name] /= gmean_speedup
            data.loc[group.index, :] = group
        
        
def join_tables(t1, t2, key=["benchmark", "exec_policy", "block_size_1d", "block_size_2d", "block_size_str",
               "size", "num_iter"], keep_common_columns=True):
    t1_tmp = t1.copy()
    t2_tmp = t2.copy()
    t1_tmp = t1_tmp.set_index(key)
    t2_tmp = t2_tmp.set_index(key)
    if keep_common_columns:
        common_columns = [x for x in t1_tmp.columns if x in t2_tmp.columns]
        t1_tmp = t1_tmp[common_columns]
        t2_tmp = t2_tmp[common_columns]

    merged = t1_tmp.merge(t2_tmp, suffixes=("_grcuda", "_cuda"), left_index=True, right_index=True, sort=True).reset_index()
    # merged = merged.merge(t2_tmp, suffixes=("_cuda2", ""), left_index=True, right_index=True, sort=True).reset_index()
    merged["grcuda_cuda_speedup"] = merged["computation_sec_cuda"] / merged["computation_sec_grcuda"]
    return merged


def join_tables_baseline(data_cuda_in, data_grcuda_in):
    data_cuda = data_cuda_in.copy()
    data_grcuda = data_grcuda_in.copy()
    baseline_policies = data_cuda["exec_policy"].unique()
    for b in baseline_policies:
        data_grcuda["speedup_" + b] = 1
    
    filter_df = ["benchmark", "block_size_str", "size", "exec_policy"]
    for k, g in data_grcuda.groupby(filter_df):
        curr_data = data_cuda[functools.reduce(np.logical_and, [data_cuda[k_b] == k_a for k_a, k_b in zip(k[:-1], filter_df[:-1])])]
        for k1, g1 in curr_data.groupby(["exec_policy"]):
            mean_exec_time = np.mean(g1["computation_sec"])
            data_grcuda.at[g.index, "speedup_" + k1] = mean_exec_time / g["computation_sec"]
    return data_grcuda


if __name__ == "__main__":
    # Outdated tests;
    input_date = "2020_06_20_20_26_03"
    data = load_data(input_date, skip_iter=3)
    
    input_date2 = "2020_06_21_14_05_38_cuda"
    data2 = load_data_cuda(input_date2, skip_iter=3)   
    
    data3 = join_tables(data[data["benchmark"] == "b1"], data2)