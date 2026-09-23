set(XPOOL_CEXT_BINDINGS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB_RECURSE XPOOL_CEXT_BINDING_SOURCES CONFIGURE_DEPENDS
  "${XPOOL_CEXT_BINDINGS_SOURCE_DIR}/*.c"
  "${XPOOL_CEXT_BINDINGS_SOURCE_DIR}/*.cc"
  "${XPOOL_CEXT_BINDINGS_SOURCE_DIR}/*.cpp"
)
file(GLOB_RECURSE XPOOL_CEXT_BINDING_HEADERS CONFIGURE_DEPENDS
  "${XPOOL_CEXT_BINDINGS_SOURCE_DIR}/*.h"
  "${XPOOL_CEXT_BINDINGS_SOURCE_DIR}/*.hh"
  "${XPOOL_CEXT_BINDINGS_SOURCE_DIR}/*.hpp"
)

if(NOT XPOOL_CEXT_BINDING_SOURCES)
  message(FATAL_ERROR "No xpool native binding sources found under ${XPOOL_CEXT_BINDINGS_SOURCE_DIR}")
endif()

Python3_add_library(
  xpool_cext MODULE WITH_SOABI
  ${XPOOL_CEXT_BINDING_SOURCES}
  ${XPOOL_CEXT_BINDING_HEADERS}
)
target_link_libraries(
  xpool_cext
  PRIVATE
    xpool_abi
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,xpool_cext_core>"
    xpool_compiler_options
    NVSHMEM::NVSHMEM
    ${TORCH_LIBRARIES}
    "${XPOOL_TORCH_PYTHON_LIBRARY}"
    CUDA::cuda_driver
    CUDA::cudart
    "${XPOOL_CUDADEVRT_LIBRARY}"
)
target_compile_features(xpool_cext PRIVATE cxx_std_20)
set_target_properties(xpool_cext PROPERTIES
  OUTPUT_NAME native
  BUILD_RPATH_USE_ORIGIN ON
  INSTALL_RPATH_USE_LINK_PATH FALSE
  INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib;$ORIGIN/../nvidia/nvshmem/lib"
)

set(XPOOL_NATIVE_STUB_DIR "${CMAKE_CURRENT_BINARY_DIR}/stubs")
set(XPOOL_NATIVE_STUB_IMPORT_DIR "${CMAKE_CURRENT_BINARY_DIR}/stubimport")
set(XPOOL_NATIVE_STUB_IMPORT_PACKAGE "${XPOOL_NATIVE_STUB_IMPORT_DIR}/xpool")
set(XPOOL_NATIVE_SOURCE_STUB_DIR "${PROJECT_SOURCE_DIR}/src/xpool/native")
# Stub generation needs a temporary importable SOABI package, then replaces the
# source-tree PEP 561 stubs as one post-link transaction.
add_custom_command(TARGET xpool_cext POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${XPOOL_NATIVE_STUB_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${XPOOL_NATIVE_STUB_IMPORT_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${XPOOL_NATIVE_STUB_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${XPOOL_NATIVE_STUB_IMPORT_PACKAGE}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${XPOOL_NATIVE_STUB_IMPORT_PACKAGE}/__init__.py"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "$<TARGET_FILE:xpool_cext>"
          "${XPOOL_NATIVE_STUB_IMPORT_PACKAGE}/$<TARGET_FILE_NAME:xpool_cext>"
  COMMAND "${CMAKE_COMMAND}" -E env
          "PYTHONPATH=${XPOOL_NATIVE_STUB_IMPORT_DIR}:${Python3_SITEARCH}"
          "${Python3_EXECUTABLE}" -S -m pybind11_stubgen
          --output-dir "${XPOOL_NATIVE_STUB_DIR}" xpool.native
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${XPOOL_NATIVE_SOURCE_STUB_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${XPOOL_NATIVE_STUB_DIR}/xpool/native"
          "${XPOOL_NATIVE_SOURCE_STUB_DIR}"
  COMMENT "Generating PEP 561 stubs for xpool.native"
  VERBATIM
)

install(TARGETS xpool_cext LIBRARY DESTINATION xpool)
install(FILES "${XPOOL_NATIVE_STUB_DIR}/xpool/native.pyi" DESTINATION xpool OPTIONAL)
install(DIRECTORY "${XPOOL_NATIVE_STUB_DIR}/xpool/native" DESTINATION xpool OPTIONAL)
