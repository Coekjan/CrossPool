cmake_minimum_required(VERSION 3.26)

set(XPOOL_CUDA_ARCHITECTURES
  "75-real;80-real;86-real;87-real;89-real;90-real;100-real;103-real;110-real;120-real;121-real;121-virtual"
  CACHE STRING "Default CUDA architectures for xpool native targets"
)
set(XPOOL_EFFECTIVE_CUDA_ARCHITECTURES "${XPOOL_CUDA_ARCHITECTURES}")
if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(XPOOL_EFFECTIVE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()
set(XPOOL_CUDA_VIRTUAL_ARCHES)
foreach(XPOOL_CUDA_ARCH IN LISTS XPOOL_EFFECTIVE_CUDA_ARCHITECTURES)
  if(XPOOL_CUDA_ARCH MATCHES "^([0-9]+)-virtual$")
    list(APPEND XPOOL_CUDA_VIRTUAL_ARCHES "${CMAKE_MATCH_1}")
  endif()
endforeach()

set(XPOOL_TORCH_CUDA_ARCH_LIST)
foreach(XPOOL_CUDA_ARCH IN LISTS XPOOL_EFFECTIVE_CUDA_ARCHITECTURES)
  if(XPOOL_CUDA_ARCH MATCHES "^([0-9]+)-real$")
    set(XPOOL_CUDA_ARCH_DIGITS "${CMAKE_MATCH_1}")
    string(LENGTH "${XPOOL_CUDA_ARCH_DIGITS}" XPOOL_CUDA_ARCH_DIGITS_LENGTH)
    math(EXPR XPOOL_CUDA_ARCH_MAJOR_LENGTH "${XPOOL_CUDA_ARCH_DIGITS_LENGTH} - 1")
    string(SUBSTRING "${XPOOL_CUDA_ARCH_DIGITS}" 0 "${XPOOL_CUDA_ARCH_MAJOR_LENGTH}" XPOOL_CUDA_ARCH_MAJOR)
    string(SUBSTRING "${XPOOL_CUDA_ARCH_DIGITS}" "${XPOOL_CUDA_ARCH_MAJOR_LENGTH}" 1 XPOOL_CUDA_ARCH_MINOR)
    set(XPOOL_TORCH_CUDA_ARCH "${XPOOL_CUDA_ARCH_MAJOR}.${XPOOL_CUDA_ARCH_MINOR}")
    list(FIND XPOOL_CUDA_VIRTUAL_ARCHES "${XPOOL_CUDA_ARCH_DIGITS}" XPOOL_CUDA_VIRTUAL_ARCH_INDEX)
    if(NOT XPOOL_CUDA_VIRTUAL_ARCH_INDEX EQUAL -1)
      string(APPEND XPOOL_TORCH_CUDA_ARCH "+PTX")
    endif()
    list(APPEND XPOOL_TORCH_CUDA_ARCH_LIST "${XPOOL_TORCH_CUDA_ARCH}")
  endif()
endforeach()

if(XPOOL_TORCH_CUDA_ARCH_LIST AND NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
  set(TORCH_CUDA_ARCH_LIST
    "${XPOOL_TORCH_CUDA_ARCH_LIST}"
    CACHE STRING "CUDA architectures passed to PyTorch's CMake integration" FORCE
  )
endif()

project(xpool LANGUAGES C CXX CUDA)

option(XPOOL_EXPORT_COMPILE_COMMANDS "Export CMake compile_commands.json" OFF)
option(XPOOL_ENABLE_CCACHE "Use ccache as a C++/CUDA compiler launcher when available" ON)
option(XPOOL_BUILD_CEXT_TESTS "Build xpool C++/CUDA unit tests" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS
  "${XPOOL_EXPORT_COMPILE_COMMANDS}"
  CACHE BOOL "Export CMake compile_commands.json" FORCE
)

if(XPOOL_ENABLE_CCACHE)
  find_program(XPOOL_CCACHE_PROGRAM ccache)
  if(XPOOL_CCACHE_PROGRAM)
    set(XPOOL_CCACHE_LANGS)
    foreach(XPOOL_CCACHE_LANG IN ITEMS C CXX CUDA)
      if(NOT CMAKE_${XPOOL_CCACHE_LANG}_COMPILER_LAUNCHER)
        set(CMAKE_${XPOOL_CCACHE_LANG}_COMPILER_LAUNCHER "${XPOOL_CCACHE_PROGRAM}")
        list(APPEND XPOOL_CCACHE_LANGS "${XPOOL_CCACHE_LANG}")
      endif()
    endforeach()
    if(XPOOL_CCACHE_LANGS)
      list(JOIN XPOOL_CCACHE_LANGS ", " XPOOL_CCACHE_LANGS_TEXT)
      message(STATUS "Using ccache compiler launcher for ${XPOOL_CCACHE_LANGS_TEXT}: ${XPOOL_CCACHE_PROGRAM}")
    else()
      message(STATUS "ccache found but compiler launchers are already configured; keeping existing launchers")
    endif()
  endif()
endif()

set(XPOOL_CEXT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cext")
set(XPOOL_CEXT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cext-include")

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

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(xpool_abi INTERFACE)
target_include_directories(xpool_abi INTERFACE ${XPOOL_CEXT_INCLUDE_DIR})
target_compile_features(xpool_abi INTERFACE cxx_std_20)

add_library(xpool_cext SHARED ${XPOOL_CEXT_SOURCES} ${XPOOL_CEXT_HEADERS})
target_link_libraries(xpool_cext PRIVATE xpool_abi ${TORCH_LIBRARIES} CUDA::cudart)
target_compile_features(xpool_cext PRIVATE cxx_std_20)
set_target_properties(xpool_cext PROPERTIES
  CUDA_ARCHITECTURES "${XPOOL_EFFECTIVE_CUDA_ARCHITECTURES}"
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  INSTALL_RPATH "$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
)

install(TARGETS xpool_cext LIBRARY DESTINATION xpool)

if(XPOOL_BUILD_CEXT_TESTS)
  include(CTest)
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
      URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)

  set(XPOOL_CEXT_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/cext")
  file(GLOB_RECURSE XPOOL_CEXT_TEST_SOURCES CONFIGURE_DEPENDS
    "${XPOOL_CEXT_TEST_DIR}/*_test.c"
    "${XPOOL_CEXT_TEST_DIR}/*_test.cc"
    "${XPOOL_CEXT_TEST_DIR}/*_test.cpp"
    "${XPOOL_CEXT_TEST_DIR}/*_test.cu"
  )

  if(NOT XPOOL_CEXT_TEST_SOURCES)
    message(FATAL_ERROR "XPOOL_BUILD_CEXT_TESTS is ON but no csrc tests were found under ${XPOOL_CEXT_TEST_DIR}")
  endif()

  foreach(XPOOL_CEXT_TEST_SOURCE IN LISTS XPOOL_CEXT_TEST_SOURCES)
    file(RELATIVE_PATH XPOOL_CEXT_TEST_REL "${XPOOL_CEXT_TEST_DIR}" "${XPOOL_CEXT_TEST_SOURCE}")
    string(REGEX REPLACE "\\.(c|cc|cpp|cu)$" "" XPOOL_CEXT_TEST_STEM "${XPOOL_CEXT_TEST_REL}")
    string(REPLACE "/" "_" XPOOL_CEXT_TEST_TARGET_SUFFIX "${XPOOL_CEXT_TEST_STEM}")
    string(REGEX REPLACE "_test$" "" XPOOL_CEXT_TEST_NAME_STEM "${XPOOL_CEXT_TEST_STEM}")
    string(REPLACE "/" "." XPOOL_CEXT_TEST_NAME_SUFFIX "${XPOOL_CEXT_TEST_NAME_STEM}")

    set(XPOOL_CEXT_TEST_TARGET "xpool_cext_${XPOOL_CEXT_TEST_TARGET_SUFFIX}")
    add_executable("${XPOOL_CEXT_TEST_TARGET}" "${XPOOL_CEXT_TEST_SOURCE}")
    target_link_libraries(
      "${XPOOL_CEXT_TEST_TARGET}"
      PRIVATE xpool_abi xpool_cext ${TORCH_LIBRARIES} CUDA::cudart GTest::gtest_main
    )
    target_include_directories("${XPOOL_CEXT_TEST_TARGET}" PRIVATE ${TORCH_INCLUDE_DIRS})
    target_compile_features("${XPOOL_CEXT_TEST_TARGET}" PRIVATE cxx_std_20)
    set_target_properties("${XPOOL_CEXT_TEST_TARGET}" PROPERTIES
      CUDA_ARCHITECTURES "${XPOOL_EFFECTIVE_CUDA_ARCHITECTURES}"
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
    )

    gtest_discover_tests("${XPOOL_CEXT_TEST_TARGET}"
      TEST_PREFIX "cext.${XPOOL_CEXT_TEST_NAME_SUFFIX}."
      DISCOVERY_TIMEOUT 30
      PROPERTIES TIMEOUT 20
    )
  endforeach()
endif()
