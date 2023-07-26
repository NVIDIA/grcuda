# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 20:46:10 2021

@author: albyr
"""

import pandas as pd
import numpy as np
import os

from compute_transfer_computation_overlap import read_nvprof_log, read_nvprof_log_a100
from load_data import DEFAULT_RES_CUDA_DIR

# V100;
GPU = "V100"
INPUT_FOLDER = "V100/nvprof_cuda/2021_10_07"

# A100;
GPU = "A100"
INPUT_FOLDER = "A100/nvprof_cuda/2021_10_18"

TRANSFERS = ["htod", "dtod", "dtoh"]

def create_nondirectional_transfer_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    devices = matrix.columns
    transfer_matrix_nondirectional = matrix + matrix.transpose()
    for i in range(len(devices)):
        for j in range(i, len(devices)):
            transfer_matrix_nondirectional.iloc[j, i] = 0
    return transfer_matrix_nondirectional

if __name__ == "__main__":
    
    res = {}
    res_summary = {}
    
    for f in os.listdir(os.path.join(DEFAULT_RES_CUDA_DIR, INPUT_FOLDER)):
        if "transfer_matrix" in f:
            continue
        print(f"reading {f}")
        if GPU == "V100":
            data = read_nvprof_log(os.path.join(DEFAULT_RES_CUDA_DIR, INPUT_FOLDER, f))
        elif GPU == "A100":
            data = read_nvprof_log_a100(os.path.join(DEFAULT_RES_CUDA_DIR, INPUT_FOLDER, f))
        else:
            raise ValueError(f"Unknown GPU={GPU}")
            
        # Keep only memory transfer;
        data = data[data["name"].isin(TRANSFERS)]
        
        data_grouped = data.groupby(["device_start", "device_end"])["transferred_data_byte"].sum().reset_index()
        devices = sorted(list(set(data_grouped["device_start"].unique()).union(set(data_grouped["device_end"].unique()))))
        transfer_matrix = np.zeros((len(devices), len(devices)))
        transfer_matrix = pd.DataFrame(transfer_matrix, index=devices, columns=devices)
        for i, r in data_grouped.iterrows():
            transfer_matrix.loc[r["device_start"]][r["device_end"]] += r["transferred_data_byte"]
        transfer_matrix_nondirectional = create_nondirectional_transfer_matrix(transfer_matrix)
        res[f] = data
        res_summary[f] = transfer_matrix_nondirectional
        
        basename = os.path.splitext(f)[0]
        basename = basename.replace("_gputrace", "").replace("m", "")
        transfer_matrix.to_csv(os.path.join(DEFAULT_RES_CUDA_DIR, INPUT_FOLDER, f"{basename}_transfer_matrix.csv"))
        