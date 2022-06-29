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
from distutils.util import strtobool

from bench.single_gpu.bench_1 import Benchmark1
from bench.single_gpu.bench_2 import Benchmark2
from bench.single_gpu.bench_3 import Benchmark3
from bench.single_gpu.bench_4 import Benchmark4
from bench.single_gpu.bench_5 import Benchmark5
from bench.single_gpu.bench_6 import Benchmark6
from bench.single_gpu.bench_7 import Benchmark7
from bench.single_gpu.bench_8 import Benchmark8
from bench.single_gpu.bench_9 import Benchmark9
from bench.single_gpu.bench_10 import Benchmark10
from bench.multi_gpu.bench_1 import Benchmark1M
from bench.multi_gpu.bench_5 import Benchmark5M
from bench.multi_gpu.bench_6 import Benchmark6M
from bench.multi_gpu.bench_9 import Benchmark9M
from bench.multi_gpu.bench_11 import Benchmark11M
from benchmark_result import BenchmarkResult

##############################
##############################

# Benchmark settings;
benchmarks = {
    # Single GPU;
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
    # Multi GPU;
    "b1m": Benchmark1M,
    "b5m": Benchmark5M,
    "b6m": Benchmark6M,
    "b9m": Benchmark9M,
    "b11m": Benchmark11M,
}

num_elem = {
    # Single GPU;
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
    # Multi GPU;
    "b1m": [100],
    "b5m": [100],
    "b6m": [100],
    "b9m": [100],
    "b11m": [100],
}

policies = {
    # Single GPU;
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
    # Multi GPU;
    "b1m": ["async"],
    "b5m": ["async"],
    "b6m": ["async"],
    "b9m": ["async"],
    "b11m": ["async"],
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
    parser.add_argument("--execution_policy",
                        help="If present, run the benchmark only with the selected execution policy")
    parser.add_argument("--dependency_policy",
                        help="If present, run the benchmark only with the selected dependency policy")
    parser.add_argument("--new_stream",
                        help="If present, run the benchmark only with the selected new stream policy")
    parser.add_argument("--parent_stream",
                        help="If present, run the benchmark only with the selected parent stream policy")
    parser.add_argument("--device_selection",
                        help="If present and parent policy is data aware, run the benchmark only with the selected device_selection")
    parser.add_argument("--memory_advise_policy",
                        help="Select a managed memory memAdvise flag, if multiple GPUs are available")
    parser.add_argument("--prefetch",
                        help="If present run the benchmark only with the selected prefetcher")
    parser.add_argument("-n", "--size", metavar="N", type=int, nargs="*",
                        help="Override the input data size used for the benchmarks")
    parser.add_argument("--number_of_gpus", metavar="N", type=int, nargs="*",
                        help="Number of GPU employed for computation")
    parser.add_argument("--block_size_1d", metavar="N", type=int, nargs="*",
                        help="Number of threads per block when using 1D kernels")
    parser.add_argument("--block_size_2d", metavar="N", type=int, nargs="*",
                        help="Number of threads per block when using 2D kernels")
    parser.add_argument("-g", "--number_of_blocks", metavar="N", type=int, nargs="?",
                        help="Number of blocks in the computation")
    parser.add_argument("-r", "--random", action="store_true",
                        help="Initialize benchmarks randomly whenever possible")
    parser.add_argument("--force_stream_attach", action="store_true",
                        help="stream_attach")
    parser.add_argument("--timing", action="store_true",
                        help="Measure the execution time of each kernel")
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
    timing = args.timing
    prefetch = args.prefetch 
    stream_attach = args.force_stream_attach
    new_stream_policy = args.new_stream
    parent_stream_policy = args.parent_stream
    device_selection = args.device_selection
    dependency_policy = args.dependency_policy
    number_of_gpus = args.number_of_gpus if args.number_of_gpus else [BenchmarkResult.DEFAULT_NUM_GPU]
    exec_policy = args.execution_policy if args.execution_policy else BenchmarkResult.DEFAULT_EXEC_POLICY
    mem_advise = args.memory_advise_policy if args.memory_advise_policy else BenchmarkResult.DEFAULT_MEM_ADVISE
    
    # Create a new benchmark result instance;
    benchmark_res = BenchmarkResult(debug=debug, num_iterations=num_iter, output_path=output_path,
                                    cpu_validation=cpu_validation, random_init=random_init)
    if benchmark_res.debug:
        BenchmarkResult.log_message(f"using CPU validation: {cpu_validation}")

    if args.benchmark:
        if benchmark_res.debug:
            BenchmarkResult.log_message(f"using only benchmark: {args.benchmark}")
        benchmarks = {b: benchmarks[b] for b in args.benchmark}

    if args.size:
        if benchmark_res.debug:
            BenchmarkResult.log_message(f"using only size: {args.size}")
        num_elem = {n: args.size for n in num_elem.keys()}

    # Setup the block size for each benchmark;
    block_sizes = BenchmarkResult.create_block_size_list(args.block_size_1d, args.block_size_2d)
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
                                benchmark.run(num_iter=i, size=n, number_of_gpus=number_of_gpus[0], block_size=block_size, exec_policy=exec_policy,
                                          dependency_policy=dependency_policy, new_stream_policy=new_stream_policy, parent_stream_policy=parent_stream_policy, device_selection=device_selection,
                                          mem_advise=mem_advise, prefetch=prefetch, stream_attach=stream_attach, timing=timing,
                                          realloc=re, reinit=ri, time_phases=time_phases, prevent_reinit=prevent_reinit,
                                          number_of_blocks=number_of_blocks)
                                prevent_reinit = True
                            # Print the summary of this block;
                            if benchmark_res.debug:
                                benchmark_res.print_current_summary(name=b_name, size=n, number_of_gpus=number_of_gpus[0], block_size=block_size, exec_policy=exec_policy,
                                          dependency_policy=dependency_policy, new_stream_policy=new_stream_policy, parent_stream_policy=parent_stream_policy, device_selection=device_selection,
                                          mem_advise=mem_advise, prefetch=prefetch, stream_attach=stream_attach, timing=timing,
                                          realloc=re, reinit=ri, time_phases=time_phases, num_blocks=number_of_blocks, skip=3)
