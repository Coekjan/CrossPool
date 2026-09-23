set(XPOOL_CEXT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(XPOOL_CEXT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src/cext-include")

file(GLOB_RECURSE XPOOL_CEXT_SOURCES CONFIGURE_DEPENDS
  "${XPOOL_CEXT_SOURCE_DIR}/*.c"
  "${XPOOL_CEXT_SOURCE_DIR}/*.cc"
  "${XPOOL_CEXT_SOURCE_DIR}/*.cpp"
  "${XPOOL_CEXT_SOURCE_DIR}/*.cu"
)
file(GLOB_RECURSE XPOOL_CEXT_HEADERS CONFIGURE_DEPENDS
  "${XPOOL_CEXT_INCLUDE_DIR}/*.h"
  "${XPOOL_CEXT_INCLUDE_DIR}/*.hh"
  "${XPOOL_CEXT_INCLUDE_DIR}/*.hpp"
  "${XPOOL_CEXT_INCLUDE_DIR}/*.cuh"
)

if(NOT XPOOL_CEXT_SOURCES)
  message(FATAL_ERROR "No XPool C++/CUDA extension sources found under ${XPOOL_CEXT_SOURCE_DIR}")
endif()

if(NOT XPOOL_CEXT_HEADERS)
  message(FATAL_ERROR "No XPool C++/CUDA extension headers found under ${XPOOL_CEXT_INCLUDE_DIR}")
endif()

add_library(xpool_abi INTERFACE)
# CUDA Toolkit installations can bundle different NVSHMEM headers; the wheel's
# headers must match the host and device libraries selected in XpoolDependencies.
target_include_directories(xpool_abi
  SYSTEM BEFORE INTERFACE ${XPOOL_NVSHMEM_INCLUDE_DIR}
)
target_include_directories(xpool_abi INTERFACE ${XPOOL_CEXT_INCLUDE_DIR})
target_link_libraries(xpool_abi INTERFACE CCCL::libcudacxx)
target_compile_features(xpool_abi INTERFACE cxx_std_20)

add_library(xpool_cext_core STATIC ${XPOOL_CEXT_SOURCES} ${XPOOL_CEXT_HEADERS})
target_link_libraries(
  xpool_cext_core
  PRIVATE
    xpool_abi
    xpool_compiler_options
    NVSHMEM::NVSHMEM
    ${TORCH_LIBRARIES}
    CUDA::cuda_driver
    CUDA::cudart
    "${XPOOL_CUDADEVRT_LIBRARY}"
)
target_compile_features(xpool_cext_core PRIVATE cxx_std_20)
set_target_properties(xpool_cext_core PROPERTIES
  # NVSHMEM device symbols are resolved during the final CUDA device link.
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)
