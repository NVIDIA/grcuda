import argparse

from bench.bench_1 import Benchmark1
from benchmark_result import BenchmarkResult

##############################
##############################

# Benchmark settings;
benchmarks = {
   "b1": Benchmark1
}

num_elem = {
   "b1": [100]
}

policies = {
    "b1": ["default"]
}

##############################
##############################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="measure GrCUDA execution time")

    parser.add_argument("-d", "--debug", action='store_true',
                        help="If present, print debug messages")
    parser.add_argument("-t", "--num_iter", metavar="N", type=int, default=BenchmarkResult.DEFAULT_NUM_ITER,
                        help="Number of times each benchmark is executed")
    parser.add_argument("-o", "--output_path", metavar="path/to/output.json",
                        help="Path to the file where results will be stored")
    parser.add_argument("--realloc", metavar="[True|False]", type=bool, nargs="*",
                        help="If True, allocate new memory and rebuild the GPU code at each iteration")
    parser.add_argument("--reinit", metavar="[True|False]", type=bool, nargs="*",
                        help="If True, re-initialize the values used in each benchmark at each iteration")
    parser.add_argument("-c", "--cpu_validation", action='store_true',
                        help="If present, validate the result of each benchmark using the CPU")
    parser.add_argument("-b", "--benchmark",
                        help="If present, run the benchmark only for the specified kernel")
    parser.add_argument("--policy",
                        help="If present, run the benchmark only with the selected policy")
    parser.add_argument("-n", "--size", metavar="N", type=int,
                        help="Override the input data size used for the benchmarks")

    # Parse the input arguments;
    args = parser.parse_args()

    debug = args.debug if args.debug else BenchmarkResult.DEFAULT_DEBUG
    num_iter = args.num_iter if args.num_iter else BenchmarkResult.DEFAULT_NUM_ITER
    output_path = args.output_path if args.output_path else ""
    cpu_validation = args.cpu_validation if args.cpu_validation else BenchmarkResult.DEFAULT_CPU_VALIDATION
    realloc = args.realloc if args.realloc else [BenchmarkResult.DEFAULT_REALLOC]
    reinit = args.reinit if args.reinit else [BenchmarkResult.DEFAULT_REINIT]

    # Create a new benchmark result instance;
    benchmark_res = BenchmarkResult(debug=debug, num_iterations=num_iter, output_path=output_path, cpu_validation=cpu_validation)

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
