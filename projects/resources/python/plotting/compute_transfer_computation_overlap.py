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
For each benchmark, we can measure how much overlap is present in the computation. We measure 4 different types of overlap:

CT, computation w.r.t transfer: percentage of GPU kernel computation that overlaps with any data transfer (host-to-device or viceversa)
TC, transfer w.r.t computation: percentage of data transfer that overlaps with (one or more) GPU kernel computation(s)
CC, computation w.r.t computation: percentage of GPU kernel computation that overlaps with any other GPU kernel computation
TOT, any type of overlap: here we consider any type of overlap between data-transfer and/or computations. Note that if a computation/data-transfer overlaps more than one computation/data-transfer, the overlap is counted only once (we consider the union of the overlap intervals)

Measures are taken for the largest data-size in the evaluation (for each benchmark), 
for the block size that results in higher speedup, to obtain a clearer understanding of what type of overlap is providing the speedup. 
In general, the TOT overlap is a good proxy of the achieved speedup, although it is sometimes inflated by high CC overlap: 
in fact, overlapping computations does not always translates to faster execution, 
especially if kernels are large enough (in terms of threads/blocks) to fill the GPU processors on their own.

Created on Mon Jul 20 10:36:56 2020

@author: alberto.parravicini
"""

import pandas as pd
import json
import os
import numpy as np
import time

DEFAULT_RES_DIR = "../../../../grcuda-data/results/scheduling_nvprof_log"

# 960
# INPUT_DATE =  "2020_10_07_960" # "2020_09_23_960"
# P100
# INPUT_DATE = "2020_10_10_P100" # "2020_09_19_P100"
# 1660
INPUT_DATE = "2020_10_10_1660" # "2020_10_06_1660"

OUTPUT_DATE = "2020_10_10"
PLOT_DIR = "../../../../grcuda-data/plots"

BENCHMARK_NAMES = {"b1": "Vector Squares", "b5": "B&S", "b6": "ML Ensemble", "b7": "HITS", "b8": "Images"}

# 960 - Unused
# DATA_DICT = {
#     "b1": "b1_31343.csv",
#     "b5": "b5_808.csv",
#     "b6": "b6_1989.csv",
#     "b7": "b7_2663.csv",
#     "b8": "b8_default_nometric_20821.csv",
#     "b10": "b10_7753.csv",
#     }

# P100 - Unused
# DATA_DICT = {
#     "b1": "b1_default_nometric_13259.csv",
#     "b5": "b5_default_nometric_14072.csv",
#     "b6": "b6_default_nometric_14724.csv",
#     "b7": "b7_default_nometric_15270.csv",
#     "b8": "b8_default_nometric_17585.csv",
#     "b10": "b10_default_nometric_17786.csv",
#     }

# 1660 - Unused
# DATA_DICT = {
#     "b1": "b1_default_nometric_True_6931.csv",
#     "b5": "b5_default_nometric_True_10628.csv",
#     "b6": "b6_default_nometric_True_14244.csv",
#     "b7": "b7_default_nometric_True_18466.csv",
#     "b8": "b8_default_nometric_True_22913.csv",
#     "b10": "b10_default_nometric_True_25151.csv",
#     }

# SKIP_SUMMARY_ROWS = {
#     "b1": 7,
#     "b5": 22,
#     "b6": 0,
#     "b7": 0,
#     "b8": 0,
#     "b10": 0,
#     }

NVPROF_HEADER = ["start_ms", "duration_ms", "Grid X", "Grid Y", "Grid Z", "Block X", "Block Y", "Block Z",
                 "Registers Per Thread"," Static SMem", "Dynamic SMem", "Device", "Context", "Stream",
                 "transferred_data_byte", "Virtual Address", "name", "Correlation_ID"]
NVPROF_HEADER_FILTERED = NVPROF_HEADER[:2] + [NVPROF_HEADER[-4]] + [NVPROF_HEADER[-2]]

OPERATIONS_TO_MERGE = set(["htod", "dtoh"])

NUM_ITER = 29

def time_phase(func: str):
    def func_call(*args, **kwargs) -> object:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return func_call

def get_overlap(a, b, c, d):
    """
    Given 2 segments (a, b) and (c, d), get the overlap length of the 2 segments;
    """
    s = max(a, c)
    e = min(b, d)
    return max(0, e - s), s, e


@time_phase
def get_overlap_ct_fast(data):
    """
    For each computation, look at the computations before it and compute the length of the overlap with them, in seconds.
    By definition, a computation has 0 overlap with itself.
    Keep only overlaps of computations with transfers;
    """
    data["overlap_ct_sec"] = 0.0
    
    segments = [(r["start_ms"], r["end_ms"], r["name"] in OPERATIONS_TO_MERGE) for i, r in data.iterrows()]
    overlap_list = np.zeros(len(segments))
    
    # Initial collection of overlaps;
    for i, row_i in enumerate(segments):
        overlaps = []
        for j, row_j in enumerate(segments):
            if row_j[0] > row_i[1]:
                break
            if i != j and not row_i[2] and row_j[2]:
                overlap, start, end = get_overlap(row_i[0], row_i[1], row_j[0], row_j[1])
                
                if overlap > 0:
                    overlaps += [(start, end)]
        overlap_list[i] = get_total_segment_set_length(overlaps)
    return sum(overlap_list), None


@time_phase    
def get_overlap_ct(data):
    """
    For each computation, look at the computations before it and compute the length of the overlap with them, in seconds.
    By definition, a computation has 0 overlap with itself.
    Keep only overlaps of computations with transfers;
    """
    data["overlap_ct_sec"] = 0.0
    overlap_matrix = np.zeros((len(data), len(data)))
    overlap_collection = set()
    # Initial collection of overlaps;
    for i, row_i in data.iterrows():
        overlaps = []
        for j, row_j in data.iterrows():
            if row_j["start_ms"] > row_i["end_ms"]:
                break
            if i != j and row_i["name"] not in OPERATIONS_TO_MERGE and row_j["name"] in OPERATIONS_TO_MERGE:
                overlap, start, end = get_overlap(row_i["start_ms"], row_i["end_ms"], row_j["start_ms"], row_j["end_ms"])
                if overlap > 0:
                    # overlap_collection.add((start, end))
                    # overlap_matrix[j, i] = overlap
                    overlaps += [(start, end)]
        data.at[i, "overlap_ct_sec"] = get_total_segment_set_length(overlaps)
    return data["overlap_ct_sec"].sum(), overlap_matrix
    # return, overlap_matrix


@time_phase
def get_overlap_tc_fast(data):
    """
    For each computation, look at the computations before it and compute the length of the overlap with them, in seconds.
    By definition, a computation has 0 overlap with itself.
    Keep only overlaps of transfers with computations;
    """
    data["overlap_tc_sec"] = 0.0
    segments = [(r["start_ms"], r["end_ms"], r["name"] in OPERATIONS_TO_MERGE) for i, r in data.iterrows()]
    overlap_list = np.zeros(len(segments))

    # Initial collection of overlaps;
    for i, row_i in enumerate(segments):
        overlaps = []
        for j, row_j in enumerate(segments):
            if row_j[0] > row_i[1]:
                break
            if i != j and row_i[2] and not row_j[2]:
                overlap, start, end = get_overlap(row_i[0], row_i[1], row_j[0], row_j[1])
                if overlap > 0:
                    overlaps += [(start, end)]
        overlap_list[i] = get_total_segment_set_length(overlaps)
    return sum(overlap_list), None


@time_phase
def get_overlap_tc(data):
    """
    For each computation, look at the computations before it and compute the length of the overlap with them, in seconds.
    By definition, a computation has 0 overlap with itself.
    Keep only overlaps of transfers with computations;
    """
    data["overlap_tc_sec"] = 0.0
    overlap_matrix = np.zeros((len(data), len(data)))
    overlap_collection = set()
    # Initial collection of overlaps;
    for i, row_i in data.iterrows():
        overlaps = []
        for j, row_j in data.iterrows():
            if row_j["start_ms"] > row_i["end_ms"]:
                break
            if i != j and row_i["name"] in OPERATIONS_TO_MERGE and row_j["name"] not in OPERATIONS_TO_MERGE:
                overlap, start, end = get_overlap(row_i["start_ms"], row_i["end_ms"], row_j["start_ms"], row_j["end_ms"])
                if overlap > 0:
                    # overlap_collection.add((start, end))
                    # overlap_matrix[j, i] = overlap
                    overlaps += [(start, end)]
        data.at[i, "overlap_tc_sec"] = get_total_segment_set_length(overlaps)
    return data["overlap_tc_sec"].sum(), overlap_matrix
    # return get_total_segment_set_length(overlap_collection), overlap_matrix



@time_phase
def get_overlap_cc_fast(data):
    """
    For each computation, look at the computations before it and compute the length of the overlap with them, in seconds.
    By definition, a computation has 0 overlap with itself.
    Keep only overlaps of computations with computations;
    """
    data["overlap_cc_sec"] = 0.0
    segments = [(r["start_ms"], r["end_ms"]) for i, r in data.iterrows() if r["name"] not in OPERATIONS_TO_MERGE]
    overlap_tot = 0

    # Initial collection of overlaps;
    for i, row_i in enumerate(segments):
        overlaps = []
        for j, row_j in enumerate(segments):
            if j >= i:
                break
            if i != j:
                overlap, start, end = get_overlap(row_i[0], row_i[1], row_j[0], row_j[1])
                if overlap > 0:
                    overlaps += [(start, end)]
        overlap_tot += get_total_segment_set_length(overlaps)
    return overlap_tot, None



@time_phase
def get_overlap_cc(data):
    """
    For each computation, look at the computations before it and compute the length of the overlap with them, in seconds.
    By definition, a computation has 0 overlap with itself.
    Keep only overlaps of computations with computations;
    """
    data["overlap_cc_sec"] = 0.0
    overlap_matrix = np.zeros((len(data), len(data)))
    overlap_collection = set()
    # Initial collection of overlaps;
    for i, row_i in data.iterrows():
        overlaps = []
        for j, row_j in data.iterrows():
            # if row_j["start_ms"] > row_i["end_ms"]:
            #     break
            if j >= i:
                break
            if i != j and row_i["name"] not in OPERATIONS_TO_MERGE and row_j["name"] not in OPERATIONS_TO_MERGE:
                overlap, start, end = get_overlap(row_i["start_ms"], row_i["end_ms"], row_j["start_ms"], row_j["end_ms"])
                if overlap > 0:
                    # overlap_collection.add((start, end))
                    # overlap_matrix[j, i] = overlap
                    overlaps += [(start, end)]
        data.at[i, "overlap_cc_sec"] = get_total_segment_set_length(overlaps)
    return data["overlap_cc_sec"].sum(), overlap_matrix
    # return get_total_segment_set_length(overlap_collection), overlap_matrix


@time_phase
def get_overlap_total_fast(data):
    """
    For each computation, look at the computations before it and compute the length of the overlap with them, in seconds.
    By definition, a computation has 0 overlap with itself;
    """
    data["overlap_tot_sec"] = 0.0
    segments = [(r["start_ms"], r["end_ms"]) for i, r in data.iterrows()]
    overlap_tot = 0

    # Initial collection of overlaps;
    for i, row_i in enumerate(segments):
        overlaps = []
        for j, row_j in enumerate(segments):
            if j >= i:
                break
            if i != j :
                overlap, start, end = get_overlap(row_i[0], row_i[1], row_j[0], row_j[1])
                if overlap > 0:
                    overlaps += [(start, end)]
        overlap_tot += get_total_segment_set_length(overlaps)
    return overlap_tot, None


@time_phase
def get_overlap_total(data):
    """
    For each computation, look at the computations before it and compute the length of the overlap with them, in seconds.
    By definition, a computation has 0 overlap with itself;
    """
    data["overlap_total_sec"] = 0.0
    overlap_matrix = np.zeros((len(data), len(data)))
    overlap_collection = set()
    # Initial collection of overlaps;
    for i, row_i in data.iterrows():
        overlaps = []
        for j, row_j in data.iterrows():
            # if row_j["start_ms"] > row_i["end_ms"]:
            #     break
            if j >= i:
                break
            if i != j:
                overlap, start, end = get_overlap(row_i["start_ms"], row_i["end_ms"], row_j["start_ms"], row_j["end_ms"])
                if overlap > 0:
                    # overlap_collection.add((start, end))
                    # overlap_matrix[j, i] = overlap
                    overlaps += [(start, end)]
        data.at[i, "overlap_total_sec"] = get_total_segment_set_length(overlaps)
    return data["overlap_total_sec"].sum(), overlap_matrix
    # return get_total_segment_set_length(overlap_collection), overlap_matrix10


def get_total_segment_set_length(segments):
    
    def merge_overlaps(a, b, c, d):
        start = max(a, c)
        end = min(b, d)
        if start < end:
            return (min(a, c), max(b, d))
        else:
            return None
    
    overlap_collection = set(segments)
    # Join overlaps until a fixed point is reached;
    while True:
        new_overlap_collection = set()
        for i, s_i in enumerate(overlap_collection):
            skip_set = set()
            merge_done = False
            for j, s_j in enumerate(overlap_collection):
                if j >= i or j in skip_set:
                    break
                overlap = merge_overlaps(*s_i, *s_j)
                # If a new merged overlap is created, add it to the collection, else add both segments;
                if overlap:
                    merge_done = True
                    skip_set.add(j)
                    new_overlap_collection.add(overlap)
            if not merge_done:
                new_overlap_collection.add(s_i)
                
        if (len(new_overlap_collection) == len(overlap_collection)) or len(overlap_collection) <= 1:
            break
        else:
           # print(len(new_overlap_collection)) 
           overlap_collection = new_overlap_collection
        
    return sum([s[1] - s[0] for s in overlap_collection])

#%%

if __name__ == "__main__":
    
    files = os.listdir(os.path.join(DEFAULT_RES_DIR, INPUT_DATE))
    files_dict = {tuple(file.split(".")[0].split("_")[:4]): file for file in files if file != "summary.csv"}
    
    # Filter files;
    filtered_files = [v for k, v in files_dict.items() if k[1] == "default" and k[2] == "nometric" and k[3] == "True"]
    
    output_res = []
    for b in filtered_files:
    
        input_file = os.path.join(DEFAULT_RES_DIR, INPUT_DATE, b)
        data = pd.read_csv(input_file, skiprows=5, names=NVPROF_HEADER)
        header = pd.read_csv(input_file, skiprows=3, nrows=1)
        start_unit = header.iloc[0, 0]
        duration_unit = header.iloc[0, 1]
        
        # Keep only a subset of columns;
        data = data[NVPROF_HEADER_FILTERED]
        
        # Remove rows with NaN Duration;
        data = data.dropna(subset=["duration_ms"]).reset_index(drop=True)
        
        # Fix data transfer column;
        data["transferred_data_byte"] = data["transferred_data_byte"].apply(lambda x: int(str(x).split(".")[0]) if not pd.isnull(x) else 0)
        
        # Convert start and duration from seconds to milliseconds;
        if start_unit == "s":
            data["start_ms"] *= 1000
        elif start_unit == "us":
            data["start_ms"] /= 1000
        if duration_unit == "s":
            data["duration_ms"] *= 1000
        elif duration_unit == "us":
            data["duration_ms"] /= 1000
        
        # Set the start of the computation equal to 0;
        data["start_ms"] -= data["start_ms"].iloc[0]
        
        # Set the end of the computation;
        data["end_ms"] = data["duration_ms"] + data["start_ms"]
        
        # Clean names of operations;
        data["name"] = data["name"].replace({
            "[Unified Memory Memcpy HtoD]": "htod",
            "[Unified Memory Memcpy DtoH]": "dtoh"
            })
        
        # Remove page faults;
        data = data[data["name"] != "[Unified Memory GPU page faults]"].reset_index(drop=True)
        data = data[data["name"] != "[Unified Memory page throttle]"].reset_index(drop=True)
        
        # Keep just the name of kernels;
        data["name"] = data["name"].apply(lambda x: x.split("(")[0])
        
        # 960
        # merge_dict = {"b1": 0.1, "b5": 0.1, "b6": 0.1, "b7": 0.1, "b8": 0.1, "b10": 0.1}
        # P100
        # merge_dict = {"b1": 0.1, "b5": 0.1, "b6": 0.1, "b7": 0.1, "b8": 0.1, "b10": 0.1}
        
        # Create a summary data-set where contiguous operations are merged;
        data["group"] = -1
        current_group = -1
        current_operation = ""
        for i, row in data.iterrows():
            tmp_operation = row["name"]
            # Keep adding to the current operation, if the time difference between the 2 operations is small enough to consider them as contiguous;
            if tmp_operation == current_operation and tmp_operation in OPERATIONS_TO_MERGE and i > 0: # and (row["start_ms"] - data.at[i - 1, "end_ms"] < merge_dict[b]):
               data.at[i, "group"] = current_group
            else:
                # New group of operations;
                current_operation = tmp_operation
                current_group += 1
                data.at[i, "group"] = current_group
                
        summary = data.groupby(["group"]).agg({
            "start_ms": np.min,
            "duration_ms": np.sum,
            "transferred_data_byte": np.sum,
            "name": lambda x: x.iloc[0],
            "end_ms": np.max,
            "group": lambda x: x.iloc[0],
            })
        
        # Ignore the first iteration;
        # summary = summary.iloc[SKIP_SUMMARY_ROWS[b]:, :].reset_index(drop=True)
        # # Set the start of the computation equal to 0;
        # summary["end_ms"] -= summary["start_ms"].iloc[0]
        # summary["start_ms"] -= summary["start_ms"].iloc[0]
        
        summary["end_ms"] = summary["duration_ms"] + summary["start_ms"]
        # Filter operations that last too little to be useful;
        # summary = summary[(summary["end_ms"] - summary["start_ms"]) > 0.001].reset_index(drop=True)
        print(b, len(summary))
        #%%
        
        # Compute 3 types of overlap: 
        # 1. Percentage of computation overlapped with transfer;
        # summary["overlap_ct_sec"] = get_overlap_matrix_ct(summary).sum(axis=0)
        # summary["overlap_ct_perc"] = summary["overlap_ct_sec"] / summary["duration_ms"]
        # ct_overlap_perc = summary[~summary["name"].isin(OPERATIONS_TO_MERGE)]["overlap_ct_perc"].mean()
        ct_overlap, ct_overlap_matrix = get_overlap_ct_fast(summary)
        ct_overlap_perc = ct_overlap / summary[~summary["name"].isin(OPERATIONS_TO_MERGE)]["duration_ms"].sum()
        
        # # 2. Percentage of transfer overlapped with computation;
        # summary["overlap_tc_sec"] = get_overlap_matrix_tc(summary).sum(axis=0) 
        # summary["overlap_tc_perc"] = summary["overlap_tc_sec"] / summary["duration_ms"]
        # tc_overlap_perc = summary[summary["name"].isin(OPERATIONS_TO_MERGE)]["overlap_tc_perc"].mean()
        tc_overlap, tc_overlap_matrix = get_overlap_tc_fast(summary)
        tc_overlap_perc = tc_overlap / summary[summary["name"].isin(OPERATIONS_TO_MERGE)]["duration_ms"].sum()
        # tc_overlap_perc = 0
        
        # # 3. Percentage of computation overlapped with other computations;
        # summary["overlap_cc_sec"] = get_overlap_matrix_cc(summary).sum(axis=0) 
        # summary["overlap_cc_perc"] = summary["overlap_cc_sec"] / summary["duration_ms"]
        # cc_overlap_perc = summary[~summary["name"].isin(OPERATIONS_TO_MERGE)]["overlap_cc_perc"].mean()
        cc_overlap, cc_overlap_matrix = get_overlap_cc_fast(summary)
        cc_overlap_perc = cc_overlap / summary[~summary["name"].isin(OPERATIONS_TO_MERGE)]["duration_ms"].sum()
        # cc_overlap_perc = 0
        
        total_overlap, total_overlap_matrix = get_overlap_total_fast(summary)
        total_overlap_perc = total_overlap / summary["duration_ms"].sum()
        # total_overlap_perc = 0
        
        print(f"Benchmark={b}; CT={100 *ct_overlap_perc:.2f}%; TC={100 * tc_overlap_perc:.2f}%; CC={100 * cc_overlap_perc:.2f}%; TOTAL={100 * total_overlap_perc:.2f}%")
        output_res += [[b.split("_")[0], ct_overlap_perc, tc_overlap_perc, cc_overlap_perc, total_overlap_perc]]
        
    # Store the DataFrame;
    out_df = pd.DataFrame(output_res)
    out_df.to_csv(os.path.join(DEFAULT_RES_DIR, INPUT_DATE, "summary.csv"), index=False, header=["benchmark", "ct_overlap_perc", "tc_overlap_perc", "cc_overlap_perc", "total_overlap_perc"])
    
    
    
    
