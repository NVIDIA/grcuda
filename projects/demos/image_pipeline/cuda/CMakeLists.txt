cmake_minimum_required(VERSION 3.16)
project(image_pipeline_cuda LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 70)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("/usr/local/include") 

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    )

add_executable(image_pipeline main.cpp opencv_interface.cpp image_pipeline.cu)
target_link_libraries(image_pipeline PRIVATE cudart ${OpenCV_LIBS})
