include_guard(GLOBAL)

set(XPOOL_LEGACY_CUDA_ARCHITECTURES
  "75-real;80-real;86-real;87-real;89-real;90-real;100-real;103-real;110-real;120-real;121-real;121-virtual"
)
set(XPOOL_CUDA_ARCHITECTURES
  "75-real;80-real;89-real;90-real;100-real;120-real"
  CACHE STRING "CUDA architectures supported by the bundled NVSHMEM device archive"
)
# Rewrite the former release default once so existing CMake caches adopt the
# architecture set supported by the bundled NVSHMEM device archive.
if(XPOOL_CUDA_ARCHITECTURES STREQUAL XPOOL_LEGACY_CUDA_ARCHITECTURES)
  set(XPOOL_CUDA_ARCHITECTURES
    "75-real;80-real;89-real;90-real;100-real;120-real"
    CACHE STRING "CUDA architectures supported by the bundled NVSHMEM device archive" FORCE
  )
endif()
set(XPOOL_EFFECTIVE_CUDA_ARCHITECTURES "${XPOOL_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_ARCHITECTURES
  "${XPOOL_EFFECTIVE_CUDA_ARCHITECTURES}"
  CACHE STRING "CUDA architectures derived from XPOOL_CUDA_ARCHITECTURES" FORCE
)
set(XPOOL_CUDA_VIRTUAL_ARCHES)
foreach(XPOOL_CUDA_ARCH IN LISTS XPOOL_EFFECTIVE_CUDA_ARCHITECTURES)
  if(XPOOL_CUDA_ARCH MATCHES "^([0-9]+)-virtual$")
    list(APPEND XPOOL_CUDA_VIRTUAL_ARCHES "${CMAKE_MATCH_1}")
  endif()
endforeach()

set(XPOOL_TORCH_CUDA_ARCH_LIST)
# PyTorch consumes its own major.minor syntax. Preserve PTX only where the CMake
# architecture list explicitly carries a matching virtual target.
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

if(XPOOL_TORCH_CUDA_ARCH_LIST)
  set(TORCH_CUDA_ARCH_LIST
    "${XPOOL_TORCH_CUDA_ARCH_LIST}"
    CACHE STRING "CUDA architectures passed to PyTorch's CMake integration" FORCE
  )
endif()
