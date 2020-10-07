from benchmark_result import BenchmarkResult
from abc import ABC, abstractmethod
from java.lang import System
from typing import Callable
import polyglot

DEFAULT_BLOCK_SIZE_1D = 32
DEFAULT_BLOCK_SIZE_2D = 8
DEFAULT_NUM_BLOCKS = 64  # GTX 960, 8 SM
DEFAULT_NUM_BLOCKS = 448  # P100, 56 SM
DEFAULT_NUM_BLOCKS = 176  # GTX 1660 Super, 22 SM

def time_phase(phase_name: str) -> Callable:
    """
    Decorator that simplifies timing a function call and storing the result in the benchmark log;
    :param phase_name: name of the benchmark phase
    :return: the output of the wrapped function
    """
    def inner_func(func) -> Callable:
        def func_call(self, *args, **kwargs) -> object:
            start = System.nanoTime()
            result = func(self, *args, **kwargs)
            end = System.nanoTime()
            self.benchmark.add_phase({"name": phase_name, "time_sec": (end - start) / 1_000_000_000})
            return result
        return func_call
    return inner_func


class Benchmark(ABC):
    """
    Base class for all benchmarks, it provides the general control flow of the benchmark execution;
    :param name: name of the benchmark
    :param benchmark: instance of BenchmarkResult, used to store results
    :param nvprof_profile: if present activate profiling for nvprof when running the benchmark
    """

    def __init__(self, name: str, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        self.name = name
        self.benchmark = benchmark
        self.nvprof_profile = nvprof_profile
        self.time_phases = False
        self.tot_iter = 0
        self.current_iter = 0
        self.random_seed = 42  # Default random seed, it will be overwritten with a random one;
        self.block_size_1d = DEFAULT_BLOCK_SIZE_1D
        self.block_size_2d = DEFAULT_BLOCK_SIZE_2D
        self.num_blocks = DEFAULT_NUM_BLOCKS
        self._block_size = {}

    @abstractmethod
    def alloc(self, size: int, block_size: dict = None) -> None:
        """
        Allocate new memory on GPU used for the benchmark;
        :param size: base factor used in the memory allocation, e.g. size of each array
        :param block_size: optional dictionary containing block size for 1D and 2D kernels
        """
        pass

    @abstractmethod
    def init(self) -> None:
        """
        Initialize the content of the input data of the benchmark;
        """
        pass

    @abstractmethod
    def reset_result(self) -> None:
        """
        Reset the values that hold the GPU result
        """
        pass

    @abstractmethod
    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:
        """
        Run an equivalent benchmark on CPU to obtain the correct result of the benchmark,
        and compute the distance w.r.t. the GPU result;
        :param gpu_result: the output of the GPU computation
        :param reinit: if the GPU data was re-initialized in this computation
        """
        pass

    @abstractmethod
    def execute(self) -> object:
        """
        Execute the main computation of this benchmark;
        :return: the result of the GPU computation, it could be a scalar numeric value or an arbitrary data structure
        """
        pass

    def execute_phase(self, phase_name, function, *args) -> object:
        """
        Executes a single step of the benchmark, possibily measuring the time it takes
        :param phase_name: name of this benchmark step
        :param function: a function to execute
        :param args: arguments of the function
        :return: the result of the function
        """
        if self.time_phases:
            start = System.nanoTime()
            res = function(*args)
            end = System.nanoTime()
            self.benchmark.add_phase({"name": phase_name, "time_sec": (end - start) / 1_000_000_000})
            return res
        else:
            return function(*args)

    def run(self, num_iter: int, policy: str, size: int, realloc: bool, reinit: bool,
            time_phases: bool, block_size: dict = None, prevent_reinit=False, number_of_blocks=DEFAULT_NUM_BLOCKS) -> None:

        # Fix missing block size;
        if "block_size_1d" not in block_size:
            block_size["block_size_1d"] = DEFAULT_BLOCK_SIZE_1D
        if "block_size_2d" not in block_size:
            block_size["block_size_2d"] = DEFAULT_BLOCK_SIZE_2D
        if number_of_blocks:
            self.num_blocks = number_of_blocks

        self.benchmark.start_new_benchmark(name=self.name,
                                           policy=policy,
                                           size=size,
                                           realloc=realloc,
                                           reinit=reinit,
                                           block_size=block_size,
                                           iteration=num_iter,
                                           time_phases=time_phases)
        self.current_iter = num_iter
        self.time_phases = time_phases
        self._block_size = block_size
        # TODO: set the execution policy;

        # Start a timer to monitor the total GPU execution time;
        start = System.nanoTime()

        # Allocate memory for the benchmark;
        if (num_iter == 0 or realloc) and not prevent_reinit:
            self.alloc(size, block_size)
        # Initialize memory for the benchmark;
        if (num_iter == 0 or reinit) and not prevent_reinit:
            self.init()

        # Reset the result;
        self.reset_result()

        # Start nvprof profiling if required;
        if self.nvprof_profile:
            polyglot.eval(language="grcuda", string="cudaProfilerStart")()

        # Execute the benchmark;
        gpu_result = self.execute()

        # Stop nvprof profiling if required;
        if self.nvprof_profile:
            polyglot.eval(language="grcuda", string="cudaProfilerStop")()

        # Stop the timer;
        end = System.nanoTime()
        self.benchmark.add_total_time((end - start) / 1_000_000_000)

        # Perform validation on CPU;
        if self.benchmark.cpu_validation:
            self.cpu_validation(gpu_result, reinit)

        # Write to file the current result;
        self.benchmark.save_to_file()
        # Book-keeping;
        self.tot_iter += 1
