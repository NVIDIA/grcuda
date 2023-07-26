# Copyright (c) 2020, 2021, 2022, NECSTLab, Politecnico di Milano. All rights reserved.

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

import argparse
import subprocess
import time
import os
from datetime import datetime
from benchmark_result import BenchmarkResult
from pathlib import Path

##############################
##############################

V100 = "V100"
A100 = "A100"
GPU = V100

BANDWIDTH_MATRIX = f"{os.getenv('GRCUDA_HOME')}/projects/resources/connection_graph/datasets/connection_graph.csv"

DEFAULT_NUM_BLOCKS = 32  # GTX 960, 8 SM
# DEFAULT_NUM_BLOCKS = 176  # GTX 1660 Super, 22 SM
# DEFAULT_NUM_BLOCKS = 448  # P100, 56 SM
if GPU == V100:
    DEFAULT_NUM_BLOCKS = 640  # V100, 80 SM
elif GPU == A100:
    DEFAULT_NUM_BLOCKS = 640

HEAP_SIZE = 26 
# HEAP_SIZE = 140 # P100
if GPU == V100:
    HEAP_SIZE = 470 # 2 x V100
elif GPU == A00:
    HEAP_SIZE = 470

# Benchmark settings;
benchmarks = [
    # Single GPU;
    # "b1",
    # "b5",
    # "b6",
    # "b7",
    # "b8",
    # "b10",
    # Multi GPU;
    "b1m",
    "b5m",
    "b6m",
    "b9m",
    "b11m",
]

# GTX 960
num_elem = {
    "b1": [20_000_000, 60_000_000, 80_000_000, 100_000_000, 120_000_000],
    "b5": [2_000_000, 6_000_000, 8_000_000, 10_000_000, 12_000_000],
    "b6": [200_000, 500_000, 800_000, 1_000_000, 1_200_000],
    "b7": [4_000_000, 7_000_000, 10_000_000, 15_000_000, 20_000_000], 
    "b8": [1600, 2400, 3200, 4000, 4800],
    "b10": [3000, 4000, 5000, 6000, 7000],
}

# GTX 1660 Super
# num_elem = {
#     "b1": [60_000_000, 80_000_000, 100_000_000, 120_000_000, 200_000_000],
#     "b5": [6_000_000, 8_000_000, 10_000_000, 12_000_000, 20_000_000],
#     "b6": [500_000, 800_000, 1_000_000, 1_200_000, 2_000_000],
#     "b7": [7_000_000, 10_000_000, 15_000_000, 20_000_000, 40_000_000],
#     "b8": [3200, 4000, 4800, 8000, 10000],
#     "b10": [6000, 7000, 10000, 12000, 14000],
# }

# P100
# num_elem = {
#     "b1": [120_000_000, 200_000_000, 500_000_000, 600_000_000, 700_000_000],
#     "b5": [12_000_000, 20_000_000, 50_000_000, 60_000_000, 70_000_000],
#     "b6": [1_200_000, 2_000_000, 4_000_000, 5_000_000, 6_000_000],
#     "b7": [20_000_000, 40_000_000, 60_000_000, 100_000_000, 140_000_000],
#     "b8": [4800, 8000, 10000, 12000, 16000],
#     "b10": [7000, 10000, 12000, 14000, 16000],
#}

# V100
if GPU == V100 or GPU == A100:
    num_elem = {
        # Single GPU;
        "b1": [160_000_000, 250_000_000, 500_000_000, 800_000_000, 950_000_000],
        "b5": [10_000_000, 16_000_000, 21_000_000, 28_000_000, 35_000_000], # out of core 50_000_000, 80_000_000, 95_000_000],
        "b6": [1_600_000, 2_500_000, 5_000_000, 6_500_000, 8_000_000],
        "b7": [25_000_000, 50_000_000, 80_000_000, 130_000_000, 180_000_000], 
        "b8": [6400, 10000, 13000, 16000, 20000],
        "b10": [12000, 16000, 18000, 20000, 22000], 
        # Multi GPU;
        "b1m": [160_000_000, 250_000_000, 500_000_000, 800_000_000, 950_000_000],
        "b5m": [10_000_000, 16_000_000, 21_000_000, 28_000_000, 35_000_000],  # out of core 50_000_000, 80_000_000, 95_000_000]
        "b6m": [1_000_000, 1_200_000, 1_400_000, 1_600_000, 1_800_000],
        "b9m": [20000, 30000, 40000, 50000, 60000],
        "b11m": [20000, 30000, 40000, 50000, 60000],
    }
# num_elem = {k: [int(v[0] / 100)] for (k, v) in num_elem.items()}  # Use this for small sizes, for debugging;

# 960
block_dim_dict = {
    # Single GPU;
    "b1": DEFAULT_NUM_BLOCKS,
    "b5": DEFAULT_NUM_BLOCKS,
    "b6": 32,
    "b7": DEFAULT_NUM_BLOCKS,
    "b8": 12,
    "b10": 16,
    "b11": DEFAULT_NUM_BLOCKS,
    # Multi GPU;
    "b1m": 64,
    "b5m": 64,
    "b6m": 64,
    "b9m": 64,
    "b11m": 64,
}

# P100
# block_dim_dict = {
#     "b1": DEFAULT_NUM_BLOCKS,
#     "b5": DEFAULT_NUM_BLOCKS,
#     "b6": 64,
#     "b7": DEFAULT_NUM_BLOCKS,
#     "b8": 32,
#     "b10": DEFAULT_NUM_BLOCKS,
#     "b11": DEFAULT_NUM_BLOCKS,
# }

# V100
if GPU == V100 or GPU == A100:
    block_dim_dict = {
        # Single GPU;
        "b1": DEFAULT_NUM_BLOCKS,
        "b5": DEFAULT_NUM_BLOCKS,
        "b6": 64,
        "b7": DEFAULT_NUM_BLOCKS,
        "b8": 32,
        "b10": DEFAULT_NUM_BLOCKS,
        "b11": DEFAULT_NUM_BLOCKS,
        # Multi GPU;
        "b1m": 64,
        "b5m": 64,
        "b6m": 64,
        "b9m": 64,
        "b11m": 64,
    }

# 1660
# block_dim_dict = {
#     "b1": DEFAULT_NUM_BLOCKS,
#     "b5": DEFAULT_NUM_BLOCKS,
#     "b6": 32,
#     "b7": DEFAULT_NUM_BLOCKS,
#     "b8": 16,
#     "b10": DEFAULT_NUM_BLOCKS,
#     "b11": DEFAULT_NUM_BLOCKS,
# }

cuda_exec_policies = ["sync", "async"]  # ["sync", "async", "cudagraph", "cudagraphmanual", "cudagraphsingle"]

exec_policies = ["sync", "async"]

dependency_policies = ["with-const"]  #, "no-const"]

new_stream_policies = ["always-new"]  #, "reuse"]

parent_stream_policies = ["disjoint", "multigpu-disjoint"]  # ["same-as-parent", "disjoint", "multigpu-early-disjoint", "multigpu-disjoint"]

choose_device_policies = ["round-robin", "stream-aware", "min-transfer-size", "minmax-transfer-time"]  # ["single-gpu", "round-robin", "stream-aware", "min-transfer-size", "minmin-transfer-time", "minmax-transfer-time"]

memory_advise = ["none"]

prefetch = ["false"]

stream_attach =  [False]

time_computation = [False]

num_gpus = [1, 2, 4, 8]

block_sizes1d_dict = {
    "b1": 32,
    "b5": 1024,
    "b6": 32,
    "b7": 32,
    "b8": 32,
    "b10": 32,
    "b11": 256, 
    # Multi GPU;
    "b1m": 32,
    "b5m": 1024,
    "b6m": 32,
    "b9m": 32,
    "b11m": 256,
}

block_sizes2d_dict = {
    "b1": 8,
    "b5": 8,
    "b6": 8,
    "b7": 8,
    "b8": 16,
    "b10": 8,
    "b11": 16,
    # Multi GPU;
    "b1m": 8,
    "b5m": 8,
    "b6m": 8,
    "b9m": 8,
    "b11m": 8,
}

##############################
##############################

CUDA_CMD = "./b -k {} -p {} -n {} -b {} -c {} -t {} -m {} -g {} {} {} | tee {}"

def execute_cuda_benchmark(benchmark, size, block_size, exec_policy, num_iter, debug, prefetch=False, stream_attach=False, num_blocks=DEFAULT_NUM_BLOCKS, num_gpus=1, output_date=None, mock=False):
    if debug:
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
        BenchmarkResult.log_message(f"benchmark={b}, size={n},"
                                    f" block size={block_size}, "
                                    f" prefetch={prefetch}, "
                                    f" num blocks={num_blocks}, "
                                    f" num GPUs={num_gpus}, "
                                    f" exec policy={exec_policy}")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")

    do_prefetch = prefetch is not None and prefetch and prefetch != "none" and prefetch != "false"

    if not output_date:
        output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"cuda_{output_date}_{benchmark}_{exec_policy}_{size}_gpu{num_gpus}_{block_size['block_size_1d']}_{block_size['block_size_2d']}_{prefetch}_{num_iter}_{num_blocks}.csv"
    # Create a folder if it doesn't exist;
    output_folder_path = os.path.join(BenchmarkResult.DEFAULT_RES_FOLDER, output_date + "_cuda")
    if not os.path.exists(output_folder_path):
        if debug:
            BenchmarkResult.log_message(f"creating result folder: {output_folder_path}")
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_folder_path, file_name)

    benchmark_cmd = CUDA_CMD.format(benchmark, exec_policy, size, block_size["block_size_1d"],
                                    block_size["block_size_2d"], num_iter, num_gpus, num_blocks, "-r" if do_prefetch else "", "-a" if stream_attach else "", output_path)
    if not mock:
        start = time.time()
        result = subprocess.run(benchmark_cmd,
                                shell=True,
                                stdout=None,
                                cwd=f"{os.getenv('GRCUDA_HOME')}/projects/resources/cuda/bin")
        result.check_returncode()
        end = time.time()
        if debug:
            BenchmarkResult.log_message(f"Benchmark total execution time: {(end - start):.2f} seconds")
    else:
        # Just print the command, for debugging;
        if debug:
            BenchmarkResult.log_message(benchmark_cmd)


##############################
##############################

GRAALPYTHON_CMD = "graalpython --vm.XX:MaxHeapSize={}G --jvm --polyglot --experimental-options " \
                  "--grcuda.ExecutionPolicy={} --grcuda.DependencyPolicy={} --grcuda.RetrieveNewStreamPolicy={} " \
                  "--grcuda.NumberOfGPUs={} --grcuda.RetrieveParentStreamPolicy={} " \
                  "--grcuda.DeviceSelectionPolicy={} --grcuda.MemAdvisePolicy={} --grcuda.InputPrefetch={} --grcuda.BandwidthMatrix={} {} {} " \
                  "benchmark_main.py -i {} -n {} -g {} --number_of_gpus {} --reinit false --realloc false " \
                  "-b {} --block_size_1d {} --block_size_2d {} --execution_policy {} --dependency_policy {} --new_stream {} "\
                  "--parent_stream {} --device_selection {} --memory_advise_policy {} --prefetch {} --no_cpu_validation {} {} {} {} -o {}"

def execute_grcuda_benchmark(benchmark, size, num_gpus, block_sizes, exec_policy, dependency_policy, new_stream_policy,
                      parent_stream_policy, choose_device_policy, memory_advise, prefetch, num_iter, bandwidth_matrix, time_phases, debug, stream_attach=False,
                      time_computation=False, num_blocks=DEFAULT_NUM_BLOCKS, output_date=None, mock=False):
    if debug:
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
        BenchmarkResult.log_message(f"benchmark={benchmark}, size={n},"
                                    f"gpus={num_gpus}, "
                                    f"block-sizes={block_sizes}, "
                                    f"num-blocks={num_blocks}, "
                                    f"exec-policy={exec_policy}, "
                                    f"dependency-policy={dependency_policy}, "
                                    f"new-stream-policy={new_stream_policy}, "
                                    f"parent-stream-policy={parent_stream_policy}, "
                                    f"choose-device-policy={choose_device_policy}, "
                                    f"mem-advise={memory_advise}, "
                                    f"prefetch={prefetch}, "
                                    f"stream-attachment={stream_attach}, "
                                    f"time-computation={time_computation}, "
                                    f"bandwidth-matrix={bandwidth_matrix}, "
                                    f"time-phases={time_phases}")
        BenchmarkResult.log_message("")

    if not output_date:
        output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{output_date}_{benchmark}_{size}_{num_gpus}_{num_blocks}_{exec_policy}_{dependency_policy}_" \
                f"{new_stream_policy}_{parent_stream_policy}_{choose_device_policy}_" \
                f"{memory_advise}_{prefetch}_{stream_attach}.json"
    # Create a folder if it doesn't exist;
    output_folder_path = os.path.join(BenchmarkResult.DEFAULT_RES_FOLDER, output_date + "_grcuda")
    if not os.path.exists(output_folder_path):
        if debug:
            BenchmarkResult.log_message(f"creating result folder: {output_folder_path}")
        if not mock:
            Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_folder_path, file_name)
    b1d_size = " ".join([str(b['block_size_1d']) for b in block_sizes])
    b2d_size = " ".join([str(b['block_size_2d']) for b in block_sizes])

    benchmark_cmd = GRAALPYTHON_CMD.format(HEAP_SIZE, exec_policy, dependency_policy, new_stream_policy,
                                           num_gpus, parent_stream_policy, choose_device_policy, memory_advise, prefetch, bandwidth_matrix,
                                           "--grcuda.ForceStreamAttach" if stream_attach else "", 
                                           "--grcuda.EnableComputationTimers" if time_computation else "",
                                           num_iter, size, num_blocks, num_gpus, benchmark, b1d_size, b2d_size, exec_policy, dependency_policy,
                                           new_stream_policy, parent_stream_policy, choose_device_policy, memory_advise, prefetch,
                                           "-d" if debug else "",
                                           "-p" if time_phases else "", 
                                           "--force_stream_attach" if stream_attach else "", 
                                           "--timing" if time_computation else "",
                                           output_path)    
    if debug:
        BenchmarkResult.log_message(benchmark_cmd)
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")
    if not mock:
        start = time.time()
        result = subprocess.run(benchmark_cmd,
                                shell=True,
                                stdout=None, #subprocess.STDOUT,
                                cwd=f"{os.getenv('GRCUDA_HOME')}/projects/resources/python/benchmark")
        result.check_returncode()
        end = time.time()
        if debug:
            BenchmarkResult.log_message(f"Benchmark total execution time: {(end - start):.2f} seconds")

##############################
##############################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wrap the GrCUDA benchmark to specify additional settings")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="If present, print debug messages")
    parser.add_argument("-c", "--cuda_test", action="store_true",
                        help="If present, run performance tests using CUDA")
    parser.add_argument("-i", "--num_iter", metavar="N", type=int, default=BenchmarkResult.DEFAULT_NUM_ITER,
                        help="Number of times each benchmark is executed")
    parser.add_argument("-g", "--num_blocks", metavar="N", type=int,
                        help="Number of blocks in each kernel, when applicable")
    parser.add_argument("-p", "--time_phases", action="store_true",
                        help="Measure the execution time of each phase of the benchmark;"
                             " note that this introduces overheads, and might influence the total execution time")
    parser.add_argument("-m", "--mock", action="store_true",
                        help="If present, simply print the benchmark CMD without executing it")
    parser.add_argument("--gpus", metavar="N", type=int, nargs="*",
                        help="Specify the maximum number of GPUs to use in the computation")

    # Parse the input arguments;
    args = parser.parse_args()

    debug = args.debug if args.debug else BenchmarkResult.DEFAULT_DEBUG
    num_iter = args.num_iter if args.num_iter else BenchmarkResult.DEFAULT_NUM_ITER
    use_cuda = args.cuda_test
    time_phases = args.time_phases
    num_blocks = args.num_blocks
    mock = args.mock
    gpus = args.gpus

    if gpus is not None:
        num_gpus = gpus

    if debug:
        BenchmarkResult.log_message(f"using block sizes: {block_sizes1d_dict} {block_sizes2d_dict}; using low-level CUDA benchmarks: {use_cuda}")

    def tot_benchmark_count():
        tot = 0
        if use_cuda:
            for b in benchmarks:
                for e in cuda_exec_policies:
                    if e == "sync":
                        tot += len(num_elem[b]) * len(prefetch) * len(stream_attach) 
                    else:
                        tot += len(num_elem[b]) * len(prefetch) * len(num_gpus) * len(stream_attach) 
        else:
            for b in benchmarks:
                for e in exec_policies:
                    if e == "sync":
                        tot += len(num_elem[b]) * len(memory_advise) * len(prefetch) * len(stream_attach) * len(time_computation)
                    else:
                        for n in num_gpus:
                            if n == 1:
                                tot += len(num_elem[b]) * len(memory_advise) * len(prefetch) * len(stream_attach) * len(time_computation)
                            else:
                                tot += len(num_elem[b]) * len(dependency_policies) * len(new_stream_policies) * len(parent_stream_policies) * len(choose_device_policies) * len(memory_advise) * len(prefetch) * len(stream_attach) * len(time_computation)
        return tot

    output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Execute each test;
    i = 0
    tot_benchmarks = tot_benchmark_count()
    for b in benchmarks:
        for n in num_elem[b]:
            if use_cuda:
                # CUDA Benchmarks;
                for e in cuda_exec_policies:
                    if e == "sync":
                        ng = [1]
                    else:
                        ng = num_gpus
                    block_sizes = BenchmarkResult.create_block_size_list([block_sizes1d_dict[b]], [block_sizes2d_dict[b]])
                    for block_size in block_sizes:
                        for p in prefetch:
                            for a in stream_attach:
                                for num_gpu in ng:
                                    nb = num_blocks if num_blocks else block_dim_dict[b]
                                    execute_cuda_benchmark(b, n, block_size, e, num_iter, debug, num_gpus=num_gpu, num_blocks=nb, prefetch=p, stream_attach=a, mock=mock, output_date=output_date)
                                    i += 1
            # GrCUDA Benchmarks;
            else:
                for exec_policy in exec_policies:
                    if exec_policy == "sync":
                        dp = [dependency_policies[0]]
                        nsp = [new_stream_policies[0]]
                        psp = [parent_stream_policies[0]]
                        cdp = [choose_device_policies[0]]
                        ng = [1]
                    else:
                        dp = dependency_policies
                        nsp = new_stream_policies
                        psp = parent_stream_policies
                        cdp = choose_device_policies
                        ng = num_gpus
                    for num_gpu in ng:
                        if exec_policy == "async" and num_gpu == 1:
                            dp = [dependency_policies[0]]
                            nsp = [new_stream_policies[0]]
                            psp = [parent_stream_policies[0]]
                            cdp = [choose_device_policies[0]]
                        else:
                            dp = dependency_policies
                            nsp = new_stream_policies
                            psp = parent_stream_policies
                            cdp = choose_device_policies
                        for m in memory_advise:
                            for p in prefetch:
                                for s in stream_attach:
                                    for t in time_computation:
                                        # Select the correct connection graph;
                                        # if GPU == V100:
                                        #     BANDWIDTH_MATRIX = f"{os.getenv('GRCUDA_HOME')}/projects/resources/connection_graph/datasets/connection_graph_{num_gpu}_v100.csv"
                                        # elif GPU == A100:
                                        #      BANDWIDTH_MATRIX = f"{os.getenv('GRCUDA_HOME')}/projects/resources/connection_graph/datasets/connection_graph_8_a100.csv"
                                        BANDWIDTH_MATRIX = f"{os.getenv('GRCUDA_HOME')}/projects/resources/connection_graph/datasets/connection_graph.csv"

                                        for dependency_policy in dp:
                                            for new_stream_policy in nsp:
                                                for parent_stream_policy in psp:
                                                    for choose_device_policy in cdp:
                                                        nb = num_blocks if num_blocks else block_dim_dict[b]
                                                        block_sizes = BenchmarkResult.create_block_size_list([block_sizes1d_dict[b]], [block_sizes2d_dict[b]])
                                                        execute_grcuda_benchmark(b, n, num_gpu, block_sizes,
                                                            exec_policy, dependency_policy, new_stream_policy, parent_stream_policy, choose_device_policy, 
                                                            m, p, num_iter, BANDWIDTH_MATRIX, time_phases, debug, s, t, nb, output_date=output_date, mock=mock)
                                                        i += 1 

