#pragma once

/// \file xpool/macros.hpp
/// \brief Cross-compiler native execution and storage annotations.

#if defined(__CUDACC__)
/// Mark a function as CUDA host-only when compiled by a CUDA compiler.
#define XPOOL_HOST_FN __host__
/// Mark a function as CUDA device-only when compiled by a CUDA compiler.
#define XPOOL_DEVICE_FN __device__
/// Mark a function as callable from both CUDA host and device code.
#define XPOOL_HOST_DEVICE_FN __host__ __device__
/// Mark a CUDA kernel entry point.
#define XPOOL_KERNEL_FN __global__
/// Place namespace-scope Device storage in CUDA constant memory.
#define XPOOL_DEVICE_CONST __device__ __constant__
/// Place Device storage in CUDA shared memory.
#define XPOOL_DEVICE_SHARED __shared__
/// Require CUDA Device function inlining.
#define XPOOL_DEVICE_FORCEINLINE __forceinline__
#else
/// Expand a CUDA annotation to nothing for an ordinary C++ compiler.
#define XPOOL_HOST_FN
#define XPOOL_HOST_DEVICE_FN
#endif
