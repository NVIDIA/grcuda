#include <string>
#include <iostream>
#include <ctime>    // For time()
#include <cstdlib>  // For srand()
#include "options.hpp"
#include "benchmark.cuh"
#include "b1.cuh"
#include "b5.cuh"
#include "b6.cuh"
#include "b7.cuh"
#include "b8.cuh"
#include "b10.cuh"

int main(int argc, char *argv[])
{

    // srand(time(0));
    srand(12);
    
    Options options = Options(argc, argv);
    BenchmarkEnum benchmark_choice = options.benchmark_choice;
    Benchmark *b;

    switch (benchmark_choice)
    {
    case BenchmarkEnum::B1:
        b = new Benchmark1(options);
        break;
    case BenchmarkEnum::B5:
        b = new Benchmark5(options);
        break;
    case BenchmarkEnum::B6:
        b = new Benchmark6(options);
        break;
    case BenchmarkEnum::B7:
        b = new Benchmark7(options);
        break;
    case BenchmarkEnum::B8:
        b = new Benchmark8(options);
        break;
    case BenchmarkEnum::B10:
        b = new Benchmark10(options);
        break;
    default:
        break;
    }
    if (b != nullptr) {
        b->run();
    } else {
        std::cout << "ERROR = benchmark is null" << std::endl;
    }
}
