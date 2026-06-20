cmake_minimum_required(VERSION 3.26)

set(XPOOL_TORCH_CUDA_ARCH_LIST
  7.5
  8.0
  8.6
  8.7
  8.9
  9.0
  10.0
  10.3
  11.0
  12.0
  12.1+PTX
)

set(XPOOL_CMAKE_CUDA_ARCHITECTURES
  75-real
  80-real
  86-real
  87-real
  89-real
  90-real
  100-real
  103-real
  110-real
  120-real
  121-real
  121-virtual
)

if(NOT DEFINED TORCH_CUDA_ARCH_LIST AND NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
  set(TORCH_CUDA_ARCH_LIST
    "${XPOOL_TORCH_CUDA_ARCH_LIST}"
    CACHE STRING "CUDA architectures passed to PyTorch's CMake integration"
  )
endif()

project(xpool LANGUAGES CXX CUDA)

option(XPOOL_ENABLE_CCACHE "Use ccache as a C++/CUDA compiler launcher when available" ON)
if(XPOOL_ENABLE_CCACHE)
  find_program(XPOOL_CCACHE_PROGRAM ccache)
endif()

set(XPOOL_CEXT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cext")
set(XPOOL_CEXT_INCLUDE_DIR "${XPOOL_CEXT_SOURCE_DIR}/include")

file(GLOB_RECURSE XPOOL_CEXT_SOURCES CONFIGURE_DEPENDS
  "${XPOOL_CEXT_SOURCE_DIR}/*.cpp"
  "${XPOOL_CEXT_SOURCE_DIR}/*.cu"
)
file(GLOB_RECURSE XPOOL_CEXT_HEADERS CONFIGURE_DEPENDS
  "${XPOOL_CEXT_INCLUDE_DIR}/*.cuh"
  "${XPOOL_CEXT_INCLUDE_DIR}/*.h"
  "${XPOOL_CEXT_INCLUDE_DIR}/*.hh"
  "${XPOOL_CEXT_INCLUDE_DIR}/*.hpp"
)

if(NOT XPOOL_CEXT_SOURCES)
  message(FATAL_ERROR "No XPool C++/CUDA extension sources found under ${XPOOL_CEXT_SOURCE_DIR}")
endif()

if(NOT XPOOL_CEXT_HEADERS)
  message(FATAL_ERROR "No XPool C++/CUDA extension headers found under ${XPOOL_CEXT_INCLUDE_DIR}")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
unset(CMAKE_CUDA_ARCHITECTURES)
unset(CMAKE_CUDA_ARCHITECTURES CACHE)
find_package(Torch CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(xpool_abi INTERFACE)
target_include_directories(xpool_abi INTERFACE ${XPOOL_CEXT_INCLUDE_DIR})
target_compile_features(xpool_abi INTERFACE cxx_std_20)

add_library(xpool_cext SHARED ${XPOOL_CEXT_SOURCES} ${XPOOL_CEXT_HEADERS})
if(XPOOL_ENABLE_CCACHE AND XPOOL_CCACHE_PROGRAM)
  set(XPOOL_CCACHE_TARGET_LANGS)
  foreach(XPOOL_CCACHE_LANG IN ITEMS CXX CUDA)
    get_target_property(XPOOL_EXISTING_LAUNCHER xpool_cext ${XPOOL_CCACHE_LANG}_COMPILER_LAUNCHER)
    if(NOT XPOOL_EXISTING_LAUNCHER)
      set_property(TARGET xpool_cext PROPERTY ${XPOOL_CCACHE_LANG}_COMPILER_LAUNCHER "${XPOOL_CCACHE_PROGRAM}")
      list(APPEND XPOOL_CCACHE_TARGET_LANGS "${XPOOL_CCACHE_LANG}")
    endif()
  endforeach()
  if(XPOOL_CCACHE_TARGET_LANGS)
    list(JOIN XPOOL_CCACHE_TARGET_LANGS ", " XPOOL_CCACHE_TARGET_LANGS_TEXT)
    message(STATUS "Using ccache compiler launcher for xpool_cext ${XPOOL_CCACHE_TARGET_LANGS_TEXT}: ${XPOOL_CCACHE_PROGRAM}")
  else()
    message(STATUS "ccache found but xpool_cext compiler launchers are already configured; keeping existing launchers")
  endif()
endif()
target_link_libraries(xpool_cext PRIVATE xpool_abi ${TORCH_LIBRARIES} CUDA::cudart)
target_compile_features(xpool_cext PRIVATE cxx_std_20)
set_target_properties(xpool_cext PROPERTIES
  CUDA_ARCHITECTURES "${XPOOL_CMAKE_CUDA_ARCHITECTURES}"
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  INSTALL_RPATH "$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
)

install(TARGETS xpool_cext LIBRARY DESTINATION xpool)
