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

DEFAULT_NUM_BLOCKS = 64

# Benchmark settings;
benchmarks = [
    # "b1",
    # "b5",
    # "b6",
    # "b7",
    # "b8",
    "b10",
]

num_elem = {
    "b1": [80_000_000],
    "b4": [40000000],
    "b5": [10_000_000],
    "b6": [800000],
    "b7": [15_000_000],
    "b8": [4800],
    "b10": [2500],
}

exec_policies = ["default", "sync"]

new_stream_policies = ["always-new"]

parent_stream_policies = ["disjoint"]

dependency_policies = ["with-const"]

block_sizes_1d_dict = {
    "b1": 32,
    "b4": 32,
    "b5": 256,
    "b6": 32,
    "b7": 32,
    "b8": 128,
    "b10": 1024,
}

block_sizes_2d_dict = {
    "b1": 8,
    "b4": 8,
    "b5": 8,
    "b6": 8,
    "b7": 8,
    "b8": 8,
    "b10": 8,
}

##############################
##############################

LOG_FOLDER = "../../../../data/nvprof_log"
METRICS = "--metrics 'dram_read_throughput,dram_write_throughput,dram_read_bytes,dram_write_bytes,l2_global_atomic_store_bytes,l2_global_load_bytes,l2_global_reduction_bytes,l2_local_global_store_bytes,l2_local_load_bytes,l2_read_throughput,l2_write_throughput,inst_executed,ipc'"

# This path is hard-coded because nvprof is executed as root,
# and the superuser doesn't have Graalpython in its environment;
GRAALPYTHON_FOLDER = "/home/users/alberto.parravicini/Documents/graalpython_venv/bin"
GRCUDA_HOME = "/home/users/alberto.parravicini/Documents/grcuda"

GRAALPYTHON_CMD = """/usr/local/cuda/bin/nvprof --csv --log-file "{}" --print-gpu-trace {} --profile-from-start off --profile-child-processes \
{}/graalpython --vm.XX:MaxHeapSize=24G --jvm --polyglot --WithThread --grcuda.RetrieveNewStreamPolicy={} \
--grcuda.ExecutionPolicy={} --grcuda.DependencyPolicy={} --grcuda.RetrieveParentStreamPolicy={} benchmark_main.py \
-i {} -n {} --reinit false --realloc false  -b {} --block_size_1d {} --block_size_2d {} --no_cpu_validation {} {} --nvprof
"""


def execute_grcuda_benchmark(benchmark, size, exec_policy, new_stream_policy,
                      parent_stream_policy, dependency_policy, num_iter, debug, time_phases):
    block_size = (block_sizes_1d_dict[b], block_sizes_2d_dict[b])
    for m in [True, False]:
        if debug:
            BenchmarkResult.log_message("")
            BenchmarkResult.log_message("")
            BenchmarkResult.log_message("#" * 30)
            BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
            BenchmarkResult.log_message(f"benchmark={b}, size={n},"
                                        f"block size={block_size}, "
                                        f"exec policy={exec_policy}, "
                                        f"new stream policy={new_stream_policy}, "
                                        f"parent stream policy={parent_stream_policy}, "
                                        f"dependency policy={dependency_policy}, "
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
        file_name = f"{b}_{exec_policy}_{'metric' if m else 'nometric'}_%p.csv"
        output_path = os.path.join(output_folder_path, file_name)


        benchmark_cmd = GRAALPYTHON_CMD.format(output_path, METRICS if m else "", GRAALPYTHON_FOLDER,
                                               new_stream_policy, exec_policy, dependency_policy, parent_stream_policy,
                                               num_iter, size, benchmark, block_size[0], block_size[1],
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
    parser.add_argument("-g", "--num_blocks", metavar="N", type=int, default=DEFAULT_NUM_BLOCKS,
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
            tot += len(num_elem[b]) * len(exec_policies)
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
                            execute_grcuda_benchmark(b, n, exec_policy, new_stream_policy,
                                                     parent_stream_policy, dependency_policy, num_iter,
                                                     debug, time_phases)
                            i += 1
