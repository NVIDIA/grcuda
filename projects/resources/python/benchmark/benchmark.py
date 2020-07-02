from benchmark_result import BenchmarkResult
from abc import ABC, abstractmethod
from java.lang import System
from typing import Callable

DEFAULT_BLOCK_SIZE_1D = 32
DEFAULT_BLOCK_SIZE_2D = 8


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
    """

    def __init__(self, name: str, benchmark: BenchmarkResult):
        self.name = name
        self.benchmark = benchmark
        self.tot_iter = 0
        self.current_iter = 0
        self.random_seed = 42  # Default random seed, it will be overwritten with a random one;
        self.block_size_1d = DEFAULT_BLOCK_SIZE_1D
        self.block_size_2d = DEFAULT_BLOCK_SIZE_2D

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

    def run(self, num_iter: int, policy: str, size: int, realloc: bool, reinit: bool, block_size: dict = None) -> None:

        # Fix missing block size;
        if "block_size_1d" not in block_size:
            block_size["block_size_1d"] = DEFAULT_BLOCK_SIZE_1D
        if "block_size_2d" not in block_size:
            block_size["block_size_2d"] = DEFAULT_BLOCK_SIZE_2D

        self.benchmark.start_new_benchmark(name=self.name,
                                           policy=policy,
                                           size=size,
                                           realloc=realloc,
                                           reinit=reinit,
                                           block_size=block_size,
                                           iteration=num_iter)
        self.current_iter = num_iter
        # TODO: set the execution policy;

        # Start a timer to monitor the total GPU execution time;
        start = System.nanoTime()

        # Allocate memory for the benchmark;
        if num_iter == 0 or realloc:
            self.alloc(size, block_size)
        # Initialize memory for the benchmark;
        if num_iter == 0 or reinit or reinit:
            self.init()

        # Reset the result;
        self.reset_result()

        # Execute the benchmark;
        gpu_result = self.execute()

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
