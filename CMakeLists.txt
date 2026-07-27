cmake_minimum_required(VERSION 3.26)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(XpoolCudaArchitectures)

project(xpool LANGUAGES C CXX CUDA)

option(XPOOL_EXPORT_COMPILE_COMMANDS "Export CMake compile_commands.json" OFF)
option(XPOOL_ENABLE_CCACHE "Use ccache as a C++/CUDA compiler launcher when available" ON)
option(XPOOL_BUILD_CEXT_TESTS "Build xpool C++/CUDA unit tests" ON)
option(XPOOL_WARNINGS_AS_ERRORS "Treat xpool native compiler warnings as errors" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS
  "${XPOOL_EXPORT_COMPILE_COMMANDS}"
  CACHE BOOL "Export CMake compile_commands.json" FORCE
)

include(XpoolCompilerCache)
include(XpoolCompilerOptions)
include(XpoolDependencies)

add_subdirectory(src/cext)
add_subdirectory(src/cext-bindings)

if(XPOOL_BUILD_CEXT_TESTS)
  include(CTest)
  add_subdirectory(tests/suites/cext)
endif()
