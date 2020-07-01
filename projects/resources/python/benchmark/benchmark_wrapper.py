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

# Benchmark settings;
benchmarks = [
    "b1",
    # "b4",
    "b6",
    "b7",
    "b8",
]

num_elem = {
    "b1": [2000000, 5000000, 10000000, 20000000, 40000000],
    "b4": [2000000, 5000000, 10000000, 20000000, 40000000],
    "b6": [20000, 50000, 200000, 500000, 800000],
    "b7": [50000, 100000, 150000, 200000, 250000],
    "b8": [800, 1600, 2400, 4000, 4800],
}

exec_policies = ["default", "sync"]

new_stream_policies = ["fifo"]

parent_stream_policies = ["disjoint"]

dependency_policies = ["with-const"]

block_sizes_1d = [32, 128, 256, 1024]
block_sizes_2d = [8, 8, 16, 32]

##############################
##############################

CUDA_CMD = "./{}_{} -n {} -b {} -c {} -t {} | tee {}"


def execute_cuda_benchmark(benchmark, size, block_size, exec_policy, num_iter, debug, output_date=None):
    if debug:
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
        BenchmarkResult.log_message(f"benchmark={b}, size={n},"
                                    f" block size={block_size}, "
                                    f" exec policy={exec_policy}")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")

    if not output_date:
        output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"cuda_{output_date}_{benchmark}_{exec_policy}_{size}_{block_size['block_size_1d']}_{block_size['block_size_2d']}_{num_iter}.csv"
    # Create a folder if it doesn't exist;
    output_folder_path = os.path.join(BenchmarkResult.DEFAULT_RES_FOLDER, output_date + "_cuda")
    if not os.path.exists(output_folder_path):
        if debug:
            BenchmarkResult.log_message(f"creating result folder: {output_folder_path}")
        os.mkdir(output_folder_path)
    output_path = os.path.join(output_folder_path, file_name)

    benchmark_cmd = CUDA_CMD.format(benchmark, exec_policy, size, block_size["block_size_1d"],
                                    block_size["block_size_2d"], num_iter, output_path)
    start = System.nanoTime()
    result = subprocess.run(benchmark_cmd,
                            shell=True,
                            stdout=subprocess.STDOUT,
                            cwd=f"{os.getenv('GRCUDA_HOME')}/projects/resources/cuda/bin")
    result.check_returncode()
    end = System.nanoTime()
    if debug:
        BenchmarkResult.log_message(f"Benchmark total execution time: {(end - start) / 1_000_000_000:.2f} seconds")


##############################
##############################

GRAALPYTHON_CMD = "graalpython --vm.XX:MaxHeapSize=24G --jvm --polyglot --WithThread " \
                  "--grcuda.RetrieveNewStreamPolicy={} --grcuda.ExecutionPolicy={} --grcuda.DependencyPolicy={} " \
                  "--grcuda.RetrieveParentStreamPolicy={} benchmark_main.py  -i {} -n {} " \
                  "--reinit false --realloc false  -b {} --block_size_1d {} --block_size_2d {} --no_cpu_validation {} -o {}"


def execute_grcuda_benchmark(benchmark, size, block_size, exec_policy, new_stream_policy,
                      parent_stream_policy, dependency_policy, num_iter, debug, output_date=None):
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

    if not output_date:
        output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{output_date}_{benchmark}_{exec_policy}_{new_stream_policy}_{parent_stream_policy}_" \
                f"{dependency_policy}_{size}_{block_size['block_size_1d']}_{block_size['block_size_2d']}_{num_iter}.json"
    # Create a folder if it doesn't exist;
    output_folder_path = os.path.join(BenchmarkResult.DEFAULT_RES_FOLDER, output_date + "_grcuda")
    if not os.path.exists(output_folder_path):
        if debug:
            BenchmarkResult.log_message(f"creating result folder: {output_folder_path}")
        os.mkdir(output_folder_path)
    output_path = os.path.join(output_folder_path, file_name)

    benchmark_cmd = GRAALPYTHON_CMD.format(new_stream_policy, exec_policy, dependency_policy, parent_stream_policy,
                                           num_iter, size, benchmark, block_size["block_size_1d"], block_size["block_size_2d"],
                                           "-d" if debug else "", output_path)
    start = System.nanoTime()
    result = subprocess.run(benchmark_cmd,
                            shell=True,
                            stdout=subprocess.STDOUT,
                            cwd=f"{os.getenv('GRCUDA_HOME')}/projects/resources/python/benchmark")
    result.check_returncode()
    end = System.nanoTime()
    if debug:
        BenchmarkResult.log_message(f"Benchmark total execution time: {(end - start) / 1_000_000_000:.2f} seconds")

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

    # Parse the input arguments;
    args = parser.parse_args()

    debug = args.debug if args.debug else BenchmarkResult.DEFAULT_DEBUG
    num_iter = args.num_iter if args.num_iter else BenchmarkResult.DEFAULT_NUM_ITER
    use_cuda = args.cuda_test

    # Setup the block size for each benchmark;
    block_sizes = create_block_size_list(block_sizes_1d, block_sizes_2d)
    if debug:
        BenchmarkResult.log_message(f"using block sizes: {block_sizes}; using low-level CUDA benchmarks: {use_cuda}")

    def tot_benchmark_count():
        tot = 0
        if use_cuda:
            for b in benchmarks:
                tot += len(num_elem[b]) * len(block_sizes) * len(exec_policies) * len(new_stream_policies) * len(parent_stream_policies) * len(dependency_policies)
        else:
            for b in benchmarks:
                tot += len(num_elem[b]) * len(block_sizes) * len(exec_policies)
        return tot

    output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Execute each test;
    i = 0
    tot_benchmarks = tot_benchmark_count()
    for b in benchmarks:
        for n in num_elem[b]:
            for block_size in block_sizes:
                for exec_policy in exec_policies:
                    # CUDA Benchmarks;
                    if use_cuda:
                        execute_cuda_benchmark(b, n, block_size, exec_policy, num_iter, debug, output_date=output_date)
                        i += 1
                    # GrCUDA Benchmarks;
                    else:
                        for new_stream_policy in new_stream_policies:
                            for parent_stream_policy in parent_stream_policies:
                                for dependency_policy in dependency_policies:
                                    execute_grcuda_benchmark(b, n, block_size, exec_policy, new_stream_policy,
                                                             parent_stream_policy, dependency_policy, num_iter, debug, output_date=output_date)
                                    i += 1
