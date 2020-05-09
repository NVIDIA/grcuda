import argparse
from distutils.util import strtobool

from bench.bench_1 import Benchmark1
from bench.bench_2 import Benchmark2
from bench.bench_3 import Benchmark3
from bench.bench_4 import Benchmark4
from bench.bench_5 import Benchmark5
from bench.bench_6 import Benchmark6
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
}

num_elem = {
    "b1": [100],
    "b2": [100],
    "b3": [100],
    "b4": [100],
    "b5": [100],
    "b6": [100],
}

policies = {
    "b1": ["default"],
    "b2": ["default"],
    "b3": ["default"],
    "b4": ["default"],
    "b5": ["default"],
    "b6": ["default"],
}

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
    parser.add_argument("-b", "--benchmark",
                        help="If present, run the benchmark only for the specified kernel")
    parser.add_argument("--policy",
                        help="If present, run the benchmark only with the selected policy")
    parser.add_argument("-n", "--size", metavar="N", type=int,
                        help="Override the input data size used for the benchmarks")
    parser.add_argument("-r", "--random", action="store_true",
                        help="Initialize benchmarks randomly whenever possible")
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

    # Create a new benchmark result instance;
    benchmark_res = BenchmarkResult(debug=debug, num_iterations=num_iter, output_path=output_path,
                                    cpu_validation=cpu_validation, random_init=random_init)

    BenchmarkResult.log_message(f"using CPU validation: {cpu_validation}")

    if args.benchmark and benchmark_res.debug:
        BenchmarkResult.log_message(f"using only benchmark: {args.benchmark}")
        benchmarks = {args.benchmark: benchmarks[args.benchmark]}

    if args.policy and benchmark_res.debug:
        BenchmarkResult.log_message(f"using only type: {args.policy}")
        policies = {n: [args.policy] for n in policies.keys()}

    if args.size and benchmark_res.debug:
        BenchmarkResult.log_message(f"using only size: {args.size}")
        num_elem = {n: [args.size] for n in num_elem.keys()}

    # Execute each test;
    for b_name, b in benchmarks.items():
        benchmark = b(benchmark_res)
        for p in policies[b_name]:
            for n in num_elem[b_name]:
                for re in realloc:
                    for ri in reinit:
                        for i in range(num_iter):
                            benchmark.run(policy=p, size=n, realloc=re, reinit=ri)
                        # Print the summary of this block;
                        if benchmark_res.debug:
                            benchmark_res.print_current_summary(name=b_name, policy=p, size=n,
                                                                realloc=re, reinit=ri, skip=3)
