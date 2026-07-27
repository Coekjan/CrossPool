#pragma once

/// \file xpool/macros.hpp
/// \brief Cross-compiler native function execution-space annotations.

#if defined(__CUDACC__)
/// Mark a function as CUDA host-only when compiled by a CUDA compiler.
#define XPOOL_HOST_FN __host__
/// Mark a function as CUDA device-only when compiled by a CUDA compiler.
#define XPOOL_DEVICE_FN __device__
/// Mark a function as callable from both CUDA host and device code.
#define XPOOL_HOST_DEVICE_FN __host__ __device__
#else
/// Expand a CUDA annotation to nothing for an ordinary C++ compiler.
#define XPOOL_HOST_FN
#define XPOOL_HOST_DEVICE_FN
#endif
