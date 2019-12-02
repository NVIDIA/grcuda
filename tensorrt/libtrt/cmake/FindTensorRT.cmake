find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_DIR} PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY NAMES nvinfer HINTS ${TENSORRT_DIR} PATH_SUFFIXES lib)

if (TENSORRT_LIBRARY AND TENSORRT_INCLUDE_DIR)
  if (NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES INTERFACE_INCLUDE_DIRECTORY "${TENSORRT_INCLUDE_DIR}")
    set_target_properties(TensorRT::nvinfer PROPERTIES
                          IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                          IMPORTED_LOCATION "${TENSORRT_LIBRARY}")
  endif()
endif()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(TensorRT
                                  REQUIRED_VARS TENSORRT_LIBRARY TENSORRT_INCLUDE_DIR)

mark_as_advanced(TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
