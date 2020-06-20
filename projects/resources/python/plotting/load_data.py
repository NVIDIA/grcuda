#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:43:46 2020

@author: alberto.parravicini
"""

import pandas as pd
import json
import os

DEFAULT_RES_DIR = "../../../../data/results"


def load_data(input_date: str) -> pd.DataFrame:
    """
    Load the benchmark results located in the input sub-folder
    :param input_date: name of the folder where results are located, as a subfolder of DEFAULT_RES_DIR
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
        benchmark, exec_policy, new_stream_policy, parent_stream_policy, dependency_policy, block_size_1d, block_size_2d = k.split("_")[6:-1]
        row += [benchmark, exec_policy, new_stream_policy, parent_stream_policy, dependency_policy, int(block_size_1d), int(block_size_2d)]

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
                            rows += [row + [num_iter, gpu_result, total_time_sec, overhead_sec, computation_sec]]

    columns = ["benchmark", "exec_policy", "new_stream_policy", "parent_stream_policy",
               "dependency_policy", "block_size_1d", "block_size_2d",
               "total_iterations", "cpu_validation", "random_init",
               "num_iter", "gpu_result", "total_time_sec", "overhead_sec", "computation_sec"]
    data = pd.DataFrame(rows, columns=columns).sort_values(by=columns[:10], ignore_index=True)
    return data


if __name__ == "__main__":
    input_date = "2020_06_20_10_53_22"
    data = load_data(input_date)
