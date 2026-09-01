include_guard(GLOBAL)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE XPOOL_TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${XPOOL_TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch CONFIG REQUIRED)
find_library(XPOOL_TORCH_PYTHON_LIBRARY
  NAMES torch_python
  HINTS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
find_package(CUDAToolkit 13.2 REQUIRED)
find_package(CCCL 3.2 CONFIG REQUIRED COMPONENTS libcudacxx
  PATHS
    "${CUDAToolkit_LIBRARY_DIR}/cmake/cccl"
    "${CUDAToolkit_LIBRARY_ROOT}/lib64/cmake/cccl"
  NO_DEFAULT_PATH
)

include(CheckCXXSourceCompiles)
# find_package() can resolve an older CCCL config from another prefix. Compile
# against the selected headers to enforce the version actually visible here.
set(CMAKE_REQUIRED_INCLUDES
  "${CUDAToolkit_INCLUDE_DIRS}"
  "${CUDAToolkit_INCLUDE_DIRS}/cccl"
)
check_cxx_source_compiles(
  "#include <cuda/version>
   #if !defined(CCCL_VERSION) || CCCL_VERSION < 3002000
   #error xpool requires CCCL 3.2 or newer
   #endif
   int main() { return 0; }"
  XPOOL_HAS_REQUIRED_CCCL
)
unset(CMAKE_REQUIRED_INCLUDES)
if(NOT XPOOL_HAS_REQUIRED_CCCL)
  message(FATAL_ERROR "xpool requires CCCL 3.2 or newer")
endif()

find_library(XPOOL_CUDADEVRT_LIBRARY
  NAMES cudadevrt libcudadevrt.a
  PATHS ${CUDAToolkit_LIBRARY_DIR}
  REQUIRED
)
# The supported NVSHMEM distribution is the Python wheel, whose headers and
# libraries live below Python3_SITELIB rather than a system installation root.
set(XPOOL_NVSHMEM_ROOT "${Python3_SITELIB}/nvidia/nvshmem")
find_path(XPOOL_NVSHMEM_INCLUDE_DIR
  NAMES nvshmem.h
  HINTS "${XPOOL_NVSHMEM_ROOT}/include"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(XPOOL_NVSHMEM_HOST_LIBRARY
  NAMES nvshmem_host libnvshmem_host.so.3
  HINTS "${XPOOL_NVSHMEM_ROOT}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(XPOOL_NVSHMEM_DEVICE_LIBRARY
  NAMES nvshmem_device libnvshmem_device.a
  HINTS "${XPOOL_NVSHMEM_ROOT}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
add_library(NVSHMEM::NVSHMEM INTERFACE IMPORTED GLOBAL)
set_target_properties(NVSHMEM::NVSHMEM PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${XPOOL_NVSHMEM_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES "${XPOOL_NVSHMEM_HOST_LIBRARY};${XPOOL_NVSHMEM_DEVICE_LIBRARY}"
  SYSTEM TRUE
)

include(FetchContent)
if(XPOOL_BUILD_CEXT_TESTS)
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      SYSTEM
      URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
      URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
  endif()
endif()
