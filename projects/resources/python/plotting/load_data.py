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

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS"" AND ANY
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
Created on Sat Jun 20 09:43:46 2020

@author: alberto.parravicini
"""

import pandas as pd
import json
import os
import numpy as np
import functools
from scipy.stats.mstats import gmean
import segretini_matplottini.src.plot_utils as pu

DEFAULT_RES_DIR = "../../../../grcuda-data/results/scheduling_multi_gpu"
DEFAULT_RES_CUDA_DIR = "../../../../grcuda-data/results/scheduling_multi_gpu"
PLOT_DIR = "../../../../grcuda-data/plots/multi_gpu"

ASYNC_POLICY_NAME = "async"   # If parsing new results;
# ASYNC_POLICY_NAME = "default"  # If parsing older results;

BENCHMARK_NAMES = {
    "b1m": "VEC",
    "b5m": "B&S",
    "b6m": "ML",
    "b9m": "CG",
    "b11m": "MUL",
    }

POLICY_NAMES = {
    "sync": "SYNC",
    ASYNC_POLICY_NAME: "ASYNC",
    }


def _load_dictionaries(input_folders: list, benchmark=""):
    dictionaries = []
    for folder in input_folders:
        input_path = os.path.join(DEFAULT_RES_DIR, folder)
    
        # Load results as JSON;
        data_dict = {}
        for res in os.listdir(input_path):
            with open(os.path.abspath(os.path.join(input_path, res)), "r") as f:
                if not benchmark or res.split("_")[6] == benchmark:
                    data_dict[res] = json.load(f)
        dictionaries += [data_dict]
    return dictionaries


def _basic_filename_cleaning(filename, dictionary):
    # Parse filename;
    benchmark = filename.split("_")[6] 
    
    # Retrieve other information;
    total_iterations = int(dictionary["num_iterations"])
    try:
        cpu_validation = dictionary["cpu_validation"].lower() == "true"
    except AttributeError:  # It's already bool;
        cpu_validation = dictionary["cpu_validation"]
    try:
        random_init = dictionary["random_init"].lower() == "true"
    except AttributeError:  # It's already bool;
        random_init = dictionary["random_init"]
    size_dict = dictionary["benchmarks"][benchmark]
    return [benchmark, total_iterations, cpu_validation, random_init], size_dict


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
    data_dict = _load_dictionaries(input_date, benchmark)
                
    phases_names = []

    # Turn results into a pd.DataFrame;
    rows = []
    for k, v in data_dict.items():
        row, size_dict = _basic_filename_cleaning(k, v)

        # Parse data for each input data size, and other settings;;
        for size, val_size in size_dict.items():
            row += [int(size)]
            for num_gpu, val_num_gpu in val_size.items():
                row += [int(num_gpu)]
                for num_blocks, val_num_blocks in val_num_gpu.items():
                    for exec_policy, val_exec_policy in val_num_blocks.items():
                        row += [exec_policy]
                        for dependency_policy, val_dependency_policy in val_exec_policy.items():
                            row += [dependency_policy]
                            for new_stream_policy, val_new_stream_policy in val_dependency_policy.items():
                                row += [new_stream_policy]
                                for parent_stream_policy, val_parent_stream_policy in val_new_stream_policy.items():
                                    row += [parent_stream_policy]
                                    for device_selection_policy, val_device_selection_policy in val_parent_stream_policy.items():
                                        row += [device_selection_policy]
                                        for prefetch, val_prefetch in val_device_selection_policy.items():
                                            row += [prefetch]
                                            for stream_attach, val_stream_attach in val_prefetch.items():
                                                row += [stream_attach.lower() == "true" or stream_attach == "True"]
                                                for kernel_timer_enabled, val_kernel_timer_enabled in val_stream_attach.items():
                                                    row += [kernel_timer_enabled == "true" or kernel_timer_enabled == "True"]
                                                    for realloc, val_realloc in val_kernel_timer_enabled.items():
                                                        row += [realloc == "true" or realloc == "True"]
                                                        for reinit, val_reinit in val_realloc.items():
                                                            row += [reinit == "true" or reinit == "True"]
                                                            for block_size, val_block_size in val_reinit.items():
                                                                # Process each iteration;
                                                                block_size_1d = int(block_size.split(",")[0])
                                                                block_size_2d = int(block_size.split(",")[1])
                                                                block_size_str = str(block_size_1d) + "," + str(block_size_2d)
                                                                row += [int(num_blocks), block_size_1d, block_size_2d, block_size_str]
                                                                
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
                                                                        rows += [row + [num_iter - skip_iter, gpu_result, total_time_sec, overhead_sec, computation_sec] + phases_time]

    columns = ["benchmark", "total_iterations", "cpu_validation", "random_init", "size", "gpus", "exec_policy", "dependency_policy", "new_stream_policy", "parent_stream_policy",
               "device_selection_policy", "prefetcher", "force_stream_attach", "kernel_timing", "realloc", "reinit",
               "num_blocks", "block_size_1d", "block_size_2d", "block_size_str", 
               "num_iter", "gpu_result", "total_time_sec", "overhead_sec", "computation_sec"] + (phases_names if phases else [])
    
    data = pd.DataFrame(rows, columns=columns).sort_values(by=columns[:20], ignore_index=True)
    
    # Clean columns with 0 computation time;
    if remove_time_zero:
        data = data[data["computation_sec"] > 0].reset_index(drop=True)
    
    # Compute speedups;
    compute_speedup(data, ["benchmark", "total_iterations", "cpu_validation", "random_init", "size", "exec_policy", "dependency_policy", "new_stream_policy",
               "device_selection_policy", "prefetcher", "force_stream_attach", "kernel_timing", "realloc", "reinit"])

    # # Clean columns with infinite speedup;
    # if remove_inf:
    #     data = data[data["computation_speedup"] != np.inf].reset_index(drop=True)
    
    return data


def load_data_grcuda_multigpu(input_folders: list, skip_iter=0, remove_inf=True, remove_time_zero=True, benchmark="", phases=None) -> pd.DataFrame:
    """
    Load the benchmark results located in the input sub-folder
    :param input_folders: list of the folders where results are located, as a subfolder of DEFAULT_RES_DIR
    :param skip_iter: skip the first iterations for each benchmark, as they are considered warmup
    :param remove_inf: remove rows with infinite speedup value, as they are not useful
    :param remove_time_zero: if True, remove rows with 0 computation time;
    :param benchmark: load data only for the specified benchmark
    :param phases: list of benchmark phases to add as columns
    :return: a DataFrame containing the results
    """
    dictionaries = _load_dictionaries(input_folders, benchmark)
    
    data_tmp = []
    for dictionary in dictionaries:
        # Turn results into a pd.DataFrame;
        rows = []
        phases_names = []
        for k, v in dictionary.items():
            row, d = _basic_filename_cleaning(k, v)
            # Parse data for each input data size, and other settings;
            for size, d in d.items():
                row += [int(size)]
                for num_gpu, d in d.items():
                    row += [int(num_gpu)]
                    for num_blocks, d in d.items():
                        for exec_policy, d in d.items():
                            row += [exec_policy]
                            for dependency_policy, d in d.items():
                                row += [dependency_policy]
                                for new_stream_policy, d in d.items():
                                    row += [new_stream_policy]
                                    for parent_stream_policy, d in d.items():
                                        row += [parent_stream_policy]
                                        for device_selection_policy, d in d.items():
                                            row += [device_selection_policy]
                                            for mem_advise, d in d.items():
                                                row += [mem_advise]
                                                for prefetch, d in d.items():
                                                    row += [prefetch]
                                                    for stream_attach, d in d.items():
                                                        row += [stream_attach.lower() == "true"]
                                                        for kernel_timer_enabled, d in d.items():
                                                            row += [kernel_timer_enabled.lower() == "true"]
                                                            for realloc, d in d.items():
                                                                row += [realloc.lower() == "true"]
                                                                for reinit, d in d.items():
                                                                    row += [reinit.lower() == "true"]
                                                                    for block_size, d in d.items():
                                                                        # Process each iteration;
                                                                        try:
                                                                            block_size_1d = int(block_size.split(",")[0])
                                                                        except:
                                                                            print(k)
                                                                        block_size_2d = int(block_size.split(",")[1])
                                                                        block_size_str = str(block_size_1d) + "," + str(block_size_2d)
                                                                        row += [int(num_blocks), block_size_1d, block_size_2d, block_size_str]
                                                                        
                                                                        for curr_iteration in d:
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
                                                                                rows += [row + [num_iter - skip_iter, gpu_result, total_time_sec, overhead_sec, computation_sec] + phases_time]
    
        columns = ["benchmark", "total_iterations", "cpu_validation", "random_init", "size", "gpus", 
                   "exec_policy", "dependency_policy", "new_stream_policy", "parent_stream_policy",
                   "device_selection_policy", "mem_advise", "prefetch", "force_stream_attach", "kernel_timing", "realloc", "reinit",
                   "num_blocks", "block_size_1d", "block_size_2d", "block_size_str", 
                   "num_iter", "gpu_result", "total_time_sec", "overhead_sec", "computation_sec"] + (phases_names if phases else [])
        
        data_tmp += [pd.DataFrame(rows, columns=columns).sort_values(by=columns[:21], ignore_index=True)]
    
    # Concatenate results;
    data = pd.concat(data_tmp, ignore_index=True)
    
    # Clean columns with 0 computation time;
    if remove_time_zero:
        data = data[data["computation_sec"] > 0].reset_index(drop=True)
        
    # FIXME: Execution time in CG ASYNC, 1 GPU explodes when using the largest size;
    data = data.query("~(benchmark == 'b9m' & exec_policy == 'async' & gpus == 1 & num_iter > 11)")
    
    # Compute speedups;
    pu.compute_speedup_df(data, key=["benchmark", "total_iterations", "cpu_validation", "random_init", "size", "dependency_policy",
                                     "mem_advise", "prefetch", "force_stream_attach", "kernel_timing", "realloc", "reinit"],
                          baseline_filter_col=["exec_policy", "new_stream_policy", "parent_stream_policy", "device_selection_policy", "gpus"],
                          baseline_filter_val=[ASYNC_POLICY_NAME, "always-new", "disjoint", "round-robin", 1],
                          time_column="computation_sec", aggregation=np.mean)
    
    # Clean columns with infinite speedup;
    if remove_inf:
        data = data[data["speedup"] != np.inf].reset_index(drop=True)
        
    data["benchmark"] = data["benchmark"].replace(BENCHMARK_NAMES)
    data["exec_policy"] = data["exec_policy"].replace(POLICY_NAMES)
    data["benchmark"] = pd.Categorical(data["benchmark"], list(BENCHMARK_NAMES.values()))
    data["exec_policy"] = pd.Categorical(data["exec_policy"], list(POLICY_NAMES.values()))
    
    data = data.sort_values(["benchmark", "exec_policy", "size", "num_iter"]).reset_index(drop=True)

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


def load_data_cuda_multigpu(input_folders: list, skip_iter=0, remove_inf=True, remove_time_zero=True) -> pd.DataFrame:
    """
    Load the benchmark results located in the input sub-folder
    :param input_folder: name of the folders where results are located, as a subfolder of DEFAULT_RES_CUDA_DIR
    :param skip_iter: skip the first iterations for each benchmark, as they are considered warmup
    :param remove_inf: if True, remove rows with infinite speedup
    :param remove_time_zero: if True, remove rows with 0 computation time;
    :return: a DataFrame containing the results
    """

    # Load results as pd.DataFrames;
    data_tmp = []
    for folder in input_folders:
        input_path = os.path.join(DEFAULT_RES_CUDA_DIR, folder)
        for f in os.listdir(input_path):
            # Parse filename;
            benchmark, exec_policy, size, num_gpu, block_size_1d, block_size_2d, prefetch, total_iterations, num_blocks = os.path.splitext(f)[0].split("_")[7:]
            tmp_data = pd.read_csv(os.path.join(input_path, f))
            
            # Skip first lines;
            tmp_data = tmp_data.iloc[skip_iter:, :]
    
            # Add other information;
            tmp_data["benchmark"] = benchmark
            tmp_data["exec_policy"] = exec_policy
            tmp_data["prefetch"] = prefetch 
            tmp_data["size"] = int(size)
            tmp_data["gpus"] = int(num_gpu.replace("gpu", ""))
            tmp_data["block_size_1d"] = int(block_size_1d)
            tmp_data["block_size_2d"] = int(block_size_2d)
            tmp_data["num_blocks"] = int(num_blocks)
            tmp_data["block_size_str"] = block_size_1d + ",8"
            tmp_data["total_iterations"] = int(total_iterations)
            data_tmp += [tmp_data]
            
    data = pd.concat(data_tmp, ignore_index=True)
    data["num_iter"] -= skip_iter
    
    # Clean names;
    data["exec_policy"].replace({"default": ASYNC_POLICY_NAME}, inplace=True)
    data["prefetch"].replace({"none": "false"}, inplace=True)

    # Reorder columns;
    columns = ["benchmark", "exec_policy", "prefetch", "block_size_1d", "block_size_2d", "num_blocks", "block_size_str",
               "total_iterations", "size", "gpus", "num_iter", "gpu_result", "total_time_sec", "overhead_sec", "computation_sec"]
    data = data[columns]
    
    # Clean columns with 0 computation time;
    if remove_time_zero:
        data = data[data["computation_sec"] > 0].reset_index(drop=True)
   
    # Compute speedups;
    pu.compute_speedup_df(data, ["benchmark", "prefetch", "block_size_1d", "block_size_2d", "size"],
                          baseline_filter_col=["exec_policy", "gpus"], baseline_filter_val=[ASYNC_POLICY_NAME, 1],
                          time_column="computation_sec")
    
    # Clean columns with infinite speedup;
    if remove_inf:
        data = data[data["speedup"] != np.inf].reset_index(drop=True)
        
    data["benchmark"] = data["benchmark"].replace(BENCHMARK_NAMES)
    data["exec_policy"] = data["exec_policy"].replace(POLICY_NAMES)
    data["benchmark"] = pd.Categorical(data["benchmark"], list(BENCHMARK_NAMES.values()))
    data["exec_policy"] = pd.Categorical(data["exec_policy"], list(POLICY_NAMES.values()))
    
    data = data.sort_values(["benchmark", "exec_policy", "size", "num_iter"]).reset_index(drop=True)
         
    return data


def compute_speedup(data, key, speedup_col_name="computation_speedup", time_column="computation_sec",
                    baseline_filter_col="gpus", baseline_filter_val=1, baseline_col_name="baseline_time_sec",
                    correction=True, aggregation=np.median):
    
    # Initialize speedup values;
    data[speedup_col_name] = 1
    data[baseline_col_name] = 0
    
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
    # input_date = "2021_10_03_12_30_18_grcuda_b5(new)_2GPU_noPrefetch_noStrAttach_allParents_dataLocality"
    # data = load_data(input_date, skip_iter=5)
    # data.to_csv("2GPU_allParents_vs_1GPU_Async.csv", sep = ';')
    
    res_list = [
        "2021_10_04_15_13_11_cuda_1gpu_v100",
        "2021_10_04_15_15_29_cuda_2gpu_v100",
        "2021_10_04_15_15_49_cuda_4gpu_v100",
        "2021_10_04_15_33_23_cuda_8gpu_v100",
        ]
    res_cuda = load_data_cuda_multigpu(res_list, skip_iter=3)
    res_cuda_grouped = res_cuda.groupby(["benchmark", "exec_policy", "num_gpu"]).mean().reset_index()
    res_cuda.to_csv(os.path.join(DEFAULT_RES_CUDA_DIR, "res_cuda.csv"), index=False)
    res_cuda_grouped.to_csv(os.path.join(DEFAULT_RES_CUDA_DIR, "res_cuda_grouped.csv"), index=False)

    # data3 = join_tables(data[data["benchmark"] == "b1"], data2)