import argparse
import subprocess
import time
import os
from datetime import datetime
from benchmark_result import BenchmarkResult
from benchmark_main import create_block_size_list

##############################
##############################

# Benchmark settings;
benchmarks = [
    "b1",
    "b6",
    "b7",
    "b8",
]

num_elem = {
    "b1": [100],
    "b6": [100],
    "b7": [100],
    "b8": [100],
}

exec_policies = ["default", "sync"]

new_stream_policies = ["fifo"]

parent_stream_policies = ["disjoint"]

dependency_policies = ["with-const"]

block_sizes_1d = [32, 1024]
block_sizes_2d = [8, 32]

##############################
##############################

GRAALPYTHON_CMD = "graalpython --vm.XX:MaxHeapSize=24G --jvm --polyglot --WithThread " \
                  "--grcuda.RetrieveNewStreamPolicy={} --grcuda.ExecutionPolicy={} --grcuda.DependencyPolicy={} " \
                  "--grcuda.RetrieveParentStreamPolicy={} benchmark_main.py  -i {} -n {} " \
                  "--reinit false --realloc false  -b {} --block_size_1d {} --block_size_2d {} --no_cpu_validation {} -o {}"


def execute_benchmark(benchmark, size, block_size, exec_policy, new_stream_policy,
                      parent_stream_policy, dependency_policy, num_iter, debug):
    if debug:
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
        BenchmarkResult.log_message(f"benchmark={b}, size={n},"
                                    f" block size={block_size}, "
                                    f" exec policy={exec_policy}, "
                                    f" new stream policy={new_stream_policy}, "
                                    f" parent stream policy={parent_stream_policy}, "
                                    f" dependency policy={dependency_policy}")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")

    output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{output_date}_{benchmark}_{exec_policy}_{new_stream_policy}_{parent_stream_policy}_{dependency_policy}_{num_iter}.json"
    output_path = os.path.join(BenchmarkResult.DEFAULT_RES_FOLDER, file_name)

    benchmark_cmd = GRAALPYTHON_CMD.format(new_stream_policy, exec_policy, dependency_policy, parent_stream_policy,
                                           num_iter, size, benchmark, block_size["block_size_1d"], block_size["block_size_2d"],
                                           "-d" if debug else "", output_path)
    start = time.time()
    result = subprocess.run(benchmark_cmd,
                            shell=True,
                            stdout=subprocess.STDOUT,
                            cwd=f"{os.getenv('GRCUDA_HOME')}/projects/resources/python/benchmark")
    result.check_returncode()
    end = time.time()
    if debug:
        BenchmarkResult.log_message(f"Benchmark total execution time: {end - start:.2f} seconds")

##############################
##############################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wrap the GrCUDA benchmark to specify additional settings")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="If present, print debug messages")
    parser.add_argument("-i", "--num_iter", metavar="N", type=int, default=BenchmarkResult.DEFAULT_NUM_ITER,
                        help="Number of times each benchmark is executed")

    # Parse the input arguments;
    args = parser.parse_args()

    debug = args.debug if args.debug else BenchmarkResult.DEFAULT_DEBUG
    num_iter = args.num_iter if args.num_iter else BenchmarkResult.DEFAULT_NUM_ITER

    # Setup the block size for each benchmark;
    block_sizes = create_block_size_list(block_sizes_1d, block_sizes_2d)
    if debug:
        BenchmarkResult.log_message(f"using block sizes: {block_sizes}")

    def tot_benchmark_count():
        tot = 0
        for b in benchmarks:
            tot += len(num_elem[b]) * len(block_sizes) * len(exec_policies) * len(new_stream_policies) * len(parent_stream_policies) * len(dependency_policies)
        return tot

    # Execute each test;
    i = 0
    tot_benchmarks = tot_benchmark_count()
    for b in benchmarks:
        for n in num_elem[b]:
            for block_size in block_sizes:
                for exec_policy in exec_policies:
                    for new_stream_policy in new_stream_policies:
                        for parent_stream_policy in parent_stream_policies:
                            for dependency_policy in dependency_policies:
                                execute_benchmark(b, n, block_size, exec_policy, new_stream_policy,
                                                  parent_stream_policy, dependency_policy, num_iter, debug)
                                i += 1
