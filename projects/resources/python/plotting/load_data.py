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

DEFAULT_RES_DIR = "../../../../data/results"


def load_data(input_date: str, skip_iter=0, remove_inf=True) -> pd.DataFrame:
    """
    Load the benchmark results located in the input sub-folder
    :param input_date: name of the folder where results are located, as a subfolder of DEFAULT_RES_DIR
    :param skip_iter: skip the first iterations for each benchmark, as they are considered warmup
    :return: a DataFrame containing the results
    """
    input_path = os.path.join(DEFAULT_RES_DIR, input_date)

    # Load results as JSON;
    data_dict = {}
    for res in os.listdir(input_path):
        with open(os.path.join(input_path, res)) as f:
            data_dict[res] = json.load(f)

    # Turn results into a pd.DataFrame;
    rows = []
    for k, v in data_dict.items():
        row = []
        # Parse filename;
        benchmark, exec_policy, new_stream_policy, parent_stream_policy, dependency_policy, _, block_size_1d, block_size_2d = k.split("_")[6:-1]
        row += [benchmark, exec_policy, new_stream_policy, parent_stream_policy, dependency_policy, int(block_size_1d), int(block_size_2d), block_size_1d + "," + block_size_2d]

        # Retrieve other information;
        total_iterations = v["num_iterations"]
        cpu_validation = v["cpu_validation"]
        random_init = v["random_init"]
        size_dict = v["benchmarks"][benchmark]["default"]
        row += [int(total_iterations), bool(cpu_validation), bool(random_init)]

        # Parse data for each input data size, and other settings;;
        for size, val_size in size_dict.items():
            for realloc, val_realloc in val_size.items():
                for reinit, val_reinit in val_realloc.items():
                    for block_size, val_block_size in val_reinit.items():
                        # Process each iteration;
                        for curr_iteration in val_block_size:
                            num_iter = curr_iteration["iteration"]
                            gpu_result = curr_iteration["gpu_result"]
                            total_time_sec = curr_iteration["total_time_sec"]
                            overhead_sec = curr_iteration["overhead_sec"]
                            computation_sec = curr_iteration["computation_sec"]
                            # Add a new row;
                            if (num_iter >= skip_iter):
                                rows += [row + [int(size), bool(realloc), bool(reinit), num_iter - skip_iter, gpu_result, total_time_sec, overhead_sec, computation_sec]]

    columns = ["benchmark", "exec_policy", "new_stream_policy", "parent_stream_policy",
               "dependency_policy", "block_size_1d", "block_size_2d", "block_size_str",
               "total_iterations", "cpu_validation", "random_init", "size", "realloc", "reinit",
               "num_iter", "gpu_result", "total_time_sec", "overhead_sec", "computation_sec"]
    data = pd.DataFrame(rows, columns=columns).sort_values(by=columns[:14], ignore_index=True)
    
    # Compute speedups;
    compute_speedup(data, ["benchmark", "new_stream_policy", "parent_stream_policy",
               "dependency_policy", "block_size_1d", "block_size_2d",
               "total_iterations", "cpu_validation", "random_init", "size", "realloc", "reinit"])
    # Clean columns with infinite speedup;
    if remove_inf:
        data = data[data["computation_speedup"] != np.inf]
    
    return data


def compute_speedup(data, key, speedup_col_name="computation_speedup", time_column="computation_sec", baseline_filter_col="exec_policy", baseline_filter_val="sync", baseline_col_name="baseline_time_sec"):
    
    # Initialize speedup values;
    data[speedup_col_name] = 1
    data[baseline_col_name] = 0
    
    grouped_data = data.groupby(key, as_index=False)
    for group_key, group in grouped_data:
        # Compute the median baseline computation time;
        median_baseline = np.median(group.loc[group[baseline_filter_col] == baseline_filter_val, time_column])
        # Compute the speedup for this group;
        data.loc[group.index, speedup_col_name] = median_baseline / group[time_column]
        data.loc[group.index, baseline_col_name] = median_baseline


if __name__ == "__main__":
    input_date = "2020_06_20_11_56_48"
    data = load_data(input_date, skip_iter=3)
