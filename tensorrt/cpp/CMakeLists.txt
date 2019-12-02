project(load_and_sample)
cmake_minimum_required(VERSION 3.12)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(TensorRT REQUIRED)

find_package(CUDA REQUIRED)

message(STATUS "TensorRT Headers: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Library: ${TENSORRT_LIBRARY}")
message(STATUS "Found CUDA ${CUDA_VERSION_STRING}")


add_executable(load_and_sample load_and_sample.cc)
target_compile_features(load_and_sample PUBLIC cxx_std_14)
target_include_directories(load_and_sample PRIVATE ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(load_and_sample PRIVATE TensorRT::nvinfer ${CUDA_LIBRARIES})
