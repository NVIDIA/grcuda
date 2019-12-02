
project(libtrt LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.10)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(TensorRT REQUIRED)

find_package(CUDA REQUIRED)

message(STATUS "TensorRT Headers: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Library: ${TENSORRT_LIBRARY}")
message(STATUS "Found CUDA ${CUDA_VERSION_STRING}")


add_library(trt SHARED src/trt.cpp)
target_compile_features(trt PUBLIC cxx_std_14)
target_include_directories(trt PUBLIC include ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(trt PRIVATE TensorRT::nvinfer ${CUDA_LIBRARIES})


add_executable(load_and_sample app/load_and_sample.c)
target_link_libraries(load_and_sample PUBLIC trt ${CUDA_LIBRARIES})




