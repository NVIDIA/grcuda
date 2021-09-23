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

import argparse
import subprocess
import time
import os
from datetime import datetime
from benchmark_result import BenchmarkResult
from benchmark_main import create_block_size_list
from java.lang import System

##############################
##############################

# True if using GPUs with capabilities with capability >= 7.5. If so, nvprof is no longer supported;
POST_TURING = True

DEFAULT_NUM_BLOCKS = 64  # GTX 960, 8 SM
DEFAULT_NUM_BLOCKS = 448  # P100, 56 SM
DEFAULT_NUM_BLOCKS = 176  # GTX 1660 Super, 22 SM

HEAP_SIZE = 26 
#HEAP_SIZE = 140 # P100

# Benchmark settings;
benchmarks = [
    "b1",
    "b5",
    "b6",
    "b7",
    "b8",
    "b10",
]


num_elem = {
    "b1": [120_000_000],
    "b5": [12_000_000],
    "b6": [1_200_000],
    "b7": [20_000_000],
    "b8": [4800],
    "b10": [7000],
}

exec_policies = ["async", "sync"]

new_stream_policies = ["always-new"]

parent_stream_policies = ["disjoint"]

dependency_policies = ["with-const"]

block_sizes_1d_dict = {
    "b1": 32,
    "b5": 256,
    "b6": 32,
    "b7": 32,
    "b8": 1024,
    "b10": 32,
}

block_sizes_2d_dict = {
    "b1": 8,
    "b5": 8,
    "b6": 8,
    "b7": 8,
    "b8": 8,
    "b10": 8,
}

block_dim_dict = {
    "b1": DEFAULT_NUM_BLOCKS,
    "b5": 32,
    "b6": DEFAULT_NUM_BLOCKS,
    "b7": DEFAULT_NUM_BLOCKS,
    "b8": 16,
    "b10": 12,
}

prefetch = [False, True]

use_metrics = [True, False]

##############################
##############################

LOG_FOLDER = "../../../../data/nvprof_log"
if POST_TURING:
    METRICS = "--metrics 'dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_atom.sum,lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_local_op_st.sum,lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum,lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_local_op_ld.sum,lts__t_sectors_op_read.sum.per_second,lts__t_sectors_op_atom.sum.per_second,lts__t_sectors_op_red.sum.per_second,lts__t_sectors_op_write.sum.per_second,lts__t_sectors_op_atom.sum.per_second,lts__t_sectors_op_red.sum.per_second,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__inst_executed.avg.per_cycle_active,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__inst_executed.sum'"
else:
    METRICS = "--metrics 'dram_read_throughput,dram_write_throughput,dram_read_bytes,dram_write_bytes,l2_global_atomic_store_bytes,l2_global_load_bytes,l2_global_reduction_bytes,l2_local_global_store_bytes,l2_local_load_bytes,l2_read_throughput,l2_write_throughput,inst_executed,ipc,flop_count_dp,flop_count_sp'"

# This path is hard-coded because nvprof is executed as root,
# and the superuser doesn't have Graalpython in its environment;
GRAALPYTHON_FOLDER = "/home/users/alberto.parravicini/Documents/graalpython_venv/bin"
GRCUDA_HOME = "/home/users/alberto.parravicini/Documents/grcuda"

if POST_TURING:
    GRAALPYTHON_CMD_METRICS = """/usr/local/cuda/bin/ncu -f --print-units base --csv --log-file "{}" --profile-from-start off --target-processes all {} \
    {}/graalpython --vm.XX:MaxHeapSize={}G --jvm --polyglot --grcuda.RetrieveNewStreamPolicy={} {} --grcuda.ForceStreamAttach \
    --grcuda.ExecutionPolicy={} --grcuda.DependencyPolicy={} --grcuda.RetrieveParentStreamPolicy={} benchmark_main.py \
    -i {} -n {} --reinit false --realloc false -g {} -b {} --block_size_1d {} --block_size_2d {} --no_cpu_validation {} {} --nvprof
    """
    GRAALPYTHON_CMD_TRACE = """/usr/local/cuda/bin/nvprof --csv --log-file "{}" --print-gpu-trace {} --profile-from-start off --profile-child-processes \
    {}/graalpython --vm.XX:MaxHeapSize={}G --jvm --polyglot --grcuda.RetrieveNewStreamPolicy={} {} --grcuda.ForceStreamAttach \
    --grcuda.ExecutionPolicy={} --grcuda.DependencyPolicy={} --grcuda.RetrieveParentStreamPolicy={} benchmark_main.py \
    -i {} -n {} --reinit false --realloc false -g {} -b {} --block_size_1d {} --block_size_2d {} --no_cpu_validation {} {} --nvprof
    """
else:
    GRAALPYTHON_CMD = """/usr/local/cuda/bin/nvprof --csv --log-file "{}" --print-gpu-trace {} --profile-from-start off --profile-child-processes \
    {}/graalpython --vm.XX:MaxHeapSize={}G --jvm --polyglot --grcuda.RetrieveNewStreamPolicy={} {} --grcuda.ForceStreamAttach \
    --grcuda.ExecutionPolicy={} --grcuda.DependencyPolicy={} --grcuda.RetrieveParentStreamPolicy={} benchmark_main.py \
    -i {} -n {} --reinit false --realloc false -g {} -b {} --block_size_1d {} --block_size_2d {} --no_cpu_validation {} {} --nvprof
    """

def execute_grcuda_benchmark(benchmark, size, exec_policy, new_stream_policy,
                      parent_stream_policy, dependency_policy, num_iter, debug, time_phases, num_blocks=DEFAULT_NUM_BLOCKS, prefetch=False):
    block_size = (block_sizes_1d_dict[b], block_sizes_2d_dict[b])
    for m in use_metrics:
        if debug:
            BenchmarkResult.log_message("")
            BenchmarkResult.log_message("")
            BenchmarkResult.log_message("#" * 30)
            BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
            BenchmarkResult.log_message(f"benchmark={b}, size={n},"
                                        f"block size={block_size}, "
                                        f"num blocks={num_blocks}, "
                                        f"exec policy={exec_policy}, "
                                        f"new stream policy={new_stream_policy}, "
                                        f"parent stream policy={parent_stream_policy}, "
                                        f"dependency policy={dependency_policy}, "
                                        f"prefetch={prefetch}, "
                                        f"time_phases={time_phases}, "
                                        f"collect metrics={m}")
            BenchmarkResult.log_message("#" * 30)
            BenchmarkResult.log_message("")
            BenchmarkResult.log_message("")

        log_folder = f"{datetime.now().strftime('%Y_%m_%d')}"
        # Create a folder if it doesn't exist;
        output_folder_path = os.path.join(LOG_FOLDER, log_folder)
        if not os.path.exists(output_folder_path):
            if debug:
                BenchmarkResult.log_message(f"creating result folder: {output_folder_path}")
            os.mkdir(output_folder_path)
        file_name = f"{b}_{exec_policy}_{'metric' if m else 'nometric'}_{prefetch}{'' if (POST_TURING and m) else '_%p'}.csv"
        output_path = os.path.join(output_folder_path, file_name)

        if POST_TURING:
            if m:
                benchmark_cmd = GRAALPYTHON_CMD_METRICS.format(output_path, METRICS, GRAALPYTHON_FOLDER, HEAP_SIZE,
                                                       new_stream_policy, "--grcuda.InputPrefetch" if prefetch else "", exec_policy, dependency_policy, parent_stream_policy,
                                                       num_iter, size, num_blocks, benchmark, block_size[0], block_size[1],
                                                       "-d" if debug else "",  "-p" if time_phases else "")
            else:
               benchmark_cmd = GRAALPYTHON_CMD_TRACE.format(output_path, "", GRAALPYTHON_FOLDER, HEAP_SIZE,
                                                   new_stream_policy, "--grcuda.InputPrefetch" if prefetch else "", exec_policy, dependency_policy, parent_stream_policy,
                                                   num_iter, size, num_blocks, benchmark, block_size[0], block_size[1],
                                                   "-d" if debug else "",  "-p" if time_phases else "") 
        else:
            benchmark_cmd = GRAALPYTHON_CMD.format(output_path, METRICS if m else "", GRAALPYTHON_FOLDER, HEAP_SIZE,
                                                   new_stream_policy, "--grcuda.InputPrefetch" if prefetch else "", exec_policy, dependency_policy, parent_stream_policy,
                                                   num_iter, size, num_blocks, benchmark, block_size[0], block_size[1],
                                                   "-d" if debug else "",  "-p" if time_phases else "")
        start = System.nanoTime()
        result = subprocess.run(benchmark_cmd,
                                shell=True,
                                stdout=subprocess.STDOUT,
                                cwd=f"{GRCUDA_HOME}/projects/resources/python/benchmark")
        result.check_returncode()
        end = System.nanoTime()
        if debug:
            BenchmarkResult.log_message(f"Benchmark total execution time: {(end - start) / 1_000_000_000:.2f} seconds")

##############################
##############################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wrap the GrCUDA benchmark to specify additional settings, and run nvprof to collect metrics")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="If present, print debug messages")
    parser.add_argument("-i", "--num_iter", metavar="N", type=int, default=BenchmarkResult.DEFAULT_NUM_ITER,
                        help="Number of times each benchmark is executed")
    parser.add_argument("-g", "--num_blocks", metavar="N", type=int,
                        help="Number of blocks in each kernel, when applicable")
    parser.add_argument("-p", "--time_phases", action="store_true",
                        help="Measure the execution time of each phase of the benchmark;"
                             " note that this introduces overheads, and might influence the total execution time")

    # Parse the input arguments;
    args = parser.parse_args()

    debug = args.debug if args.debug else BenchmarkResult.DEFAULT_DEBUG
    num_iter = args.num_iter if args.num_iter else BenchmarkResult.DEFAULT_NUM_ITER
    time_phases = args.time_phases
    num_blocks = args.num_blocks

    def tot_benchmark_count():
        tot = 0
        for b in benchmarks:
            tot += len(num_elem[b]) * len(exec_policies) * len(prefetch) * len(use_metrics)
        return tot

    output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Execute each test;
    i = 0
    tot_benchmarks = tot_benchmark_count()
    for b in benchmarks:
        for n in num_elem[b]:
            for exec_policy in exec_policies:
                # GrCUDA Benchmarks;
                for new_stream_policy in new_stream_policies:
                    for parent_stream_policy in parent_stream_policies:
                        for dependency_policy in dependency_policies:
                            for p in prefetch:
                                nb = num_blocks if num_blocks else block_dim_dict[b]
                                execute_grcuda_benchmark(b, n, exec_policy, new_stream_policy,
                                                         parent_stream_policy, dependency_policy, num_iter,
                                                         debug, time_phases, num_blocks=nb, prefetch=p)
                                i += 1
