import argparse
from distutils.util import strtobool

from bench.bench_1 import Benchmark1
from bench.bench_2 import Benchmark2
from bench.bench_3 import Benchmark3
from bench.bench_4 import Benchmark4
from bench.bench_5 import Benchmark5
from bench.bench_6 import Benchmark6
from bench.bench_72 import Benchmark7
from bench.bench_8 import Benchmark8
from bench.bench_9 import Benchmark9
from bench.bench_10 import Benchmark10
from benchmark_result import BenchmarkResult

##############################
##############################

# Benchmark settings;
benchmarks = {
    "b1": Benchmark1,
    "b2": Benchmark2,
    "b3": Benchmark3,
    "b4": Benchmark4,
    "b5": Benchmark5,
    "b6": Benchmark6,
    "b7": Benchmark7,
    "b8": Benchmark8,
    "b9": Benchmark9,
    "b10": Benchmark10,
}

num_elem = {
    "b1": [100],
    "b2": [100],
    "b3": [100],
    "b4": [100],
    "b5": [100],
    "b6": [100],
    "b7": [100],
    "b8": [100],
    "b9": [100],
    "b10": [100],
}

policies = {
    "b1": ["async"],
    "b2": ["async"],
    "b3": ["async"],
    "b4": ["async"],
    "b5": ["async"],
    "b6": ["async"],
    "b7": ["async"],
    "b8": ["async"],
    "b9": ["async"],
    "b10": ["async"],
}

##############################
##############################


def create_block_size_list(block_size_1d, block_size_2d) -> list:
    if (not block_size_1d) and block_size_2d:  # Only 2D block size;
        block_size = [{"block_size_2d": b} for b in block_size_2d]
    elif (not block_size_2d) and block_size_1d:  # Only 1D block size;
        block_size = [{"block_size_1d": b} for b in block_size_1d]
    elif block_size_1d and block_size_2d:  # Both 1D and 2D size;
        # Ensure they have the same size;
        if len(block_size_2d) > len(block_size_1d):
            block_size_1d = block_size_1d + [block_size_1d[-1]] * (len(block_size_2d) - len(block_size_1d))
        elif len(block_size_1d) > len(block_size_2d):
            block_size_2d = block_size_2d + [block_size_2d[-1]] * (len(block_size_1d) - len(block_size_2d))
        block_size = [{"block_size_1d": x[0], "block_size_2d": x[1]} for x in zip(block_size_1d, block_size_2d)]
    else:
        block_size = [{}]
    return block_size

##############################
##############################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="measure GrCUDA execution time")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="If present, print debug messages")
    parser.add_argument("-i", "--num_iter", metavar="N", type=int, default=BenchmarkResult.DEFAULT_NUM_ITER,
                        help="Number of times each benchmark is executed")
    parser.add_argument("-o", "--output_path", metavar="path/to/output.json",
                        help="Path to the file where results will be stored")
    parser.add_argument("--realloc", metavar="[True|False]", type=lambda x: bool(strtobool(x)), nargs="*",
                        help="If True, allocate new memory and rebuild the GPU code at each iteration")
    parser.add_argument("--reinit", metavar="[True|False]", type=lambda x: bool(strtobool(x)), nargs="*",
                        help="If True, re-initialize the values used in each benchmark at each iteration")
    parser.add_argument("-c", "--cpu_validation", action="store_true", dest="cpu_validation",
                        help="Validate the result of each benchmark using the CPU")
    parser.add_argument("--no_cpu_validation", action="store_false", dest="cpu_validation",
                        help="Validate the result of each benchmark using the CPU")
    parser.add_argument("-b", "--benchmark", nargs="*",
                        help="If present, run the benchmark only for the specified kernel")
    parser.add_argument("--policy",
                        help="If present, run the benchmark only with the selected policy")
    parser.add_argument("-n", "--size", metavar="N", type=int, nargs="*",
                        help="Override the input data size used for the benchmarks")
    parser.add_argument("--block_size_1d", metavar="N", type=int, nargs="*",
                        help="Number of threads per block when using 1D kernels")
    parser.add_argument("--block_size_2d", metavar="N", type=int, nargs="*",
                        help="Number of threads per block when using 2D kernels")
    parser.add_argument("-g", "--number_of_blocks", metavar="N", type=int, nargs="?",
                        help="Number of blocks in the computation")
    parser.add_argument("-r", "--random", action="store_true",
                        help="Initialize benchmarks randomly whenever possible")
    parser.add_argument("-p", "--time_phases", action="store_true",
                        help="Measure the execution time of each phase of the benchmark;"
                             " note that this introduces overheads, and might influence the total execution time")
    parser.add_argument("--nvprof", action="store_true",
                        help="If present, enable profiling when using nvprof."
                             " For this option to have effect, run graalpython using nvprof, with flag '--profile-from-start off'")
    parser.set_defaults(cpu_validation=BenchmarkResult.DEFAULT_CPU_VALIDATION)

    # Parse the input arguments;
    args = parser.parse_args()

    debug = args.debug if args.debug else BenchmarkResult.DEFAULT_DEBUG
    num_iter = args.num_iter if args.num_iter else BenchmarkResult.DEFAULT_NUM_ITER
    output_path = args.output_path if args.output_path else ""
    realloc = args.realloc if args.realloc else [BenchmarkResult.DEFAULT_REALLOC]
    reinit = args.reinit if args.reinit else [BenchmarkResult.DEFAULT_REINIT]
    random_init = args.random if args.random else BenchmarkResult.DEFAULT_RANDOM_INIT
    cpu_validation = args.cpu_validation
    time_phases = args.time_phases
    nvprof_profile = args.nvprof

    # Create a new benchmark result instance;
    benchmark_res = BenchmarkResult(debug=debug, num_iterations=num_iter, output_path=output_path,
                                    cpu_validation=cpu_validation, random_init=random_init)
    if benchmark_res.debug:
        BenchmarkResult.log_message(f"using CPU validation: {cpu_validation}")

    if args.benchmark:
        if benchmark_res.debug:
            BenchmarkResult.log_message(f"using only benchmark: {args.benchmark}")
        benchmarks = {b: benchmarks[b] for b in args.benchmark}

    if args.policy:
        if benchmark_res.debug:
            BenchmarkResult.log_message(f"using only type: {args.policy}")
        policies = {n: [args.policy] for n in policies.keys()}

    if args.size:
        if benchmark_res.debug:
            BenchmarkResult.log_message(f"using only size: {args.size}")
        num_elem = {n: args.size for n in num_elem.keys()}

    # Setup the block size for each benchmark;
    block_sizes = create_block_size_list(args.block_size_1d, args.block_size_2d)
    number_of_blocks = args.number_of_blocks
    if (args.block_size_1d or args.block_size_2d) and benchmark_res.debug:
        BenchmarkResult.log_message(f"using block sizes: {block_sizes}")
    if number_of_blocks:
        BenchmarkResult.log_message(f"using number of blocks: {number_of_blocks}")

    # Execute each test;
    for b_name, b in benchmarks.items():
        benchmark = b(benchmark_res, nvprof_profile=nvprof_profile)
        for p in policies[b_name]:
            for n in num_elem[b_name]:
                prevent_reinit = False
                for re in realloc:
                    for ri in reinit:
                        for block_size in block_sizes:
                            for i in range(num_iter):
                                benchmark.run(num_iter=i, policy=p, size=n, realloc=re, reinit=ri,
                                              block_size=block_size, time_phases=time_phases, prevent_reinit=prevent_reinit, number_of_blocks=number_of_blocks)
                                prevent_reinit = True
                            # Print the summary of this block;
                            if benchmark_res.debug:
                                benchmark_res.print_current_summary(name=b_name, policy=p, size=n,
                                                                    realloc=re, reinit=ri, block_size=block_size, skip=3)
