#pragma once

/// \file xpool/utils/device.cuh
/// \brief Device-side CUDA utility primitives.

#include <crt/device_functions.h>

namespace xpool::utils::device {

/// Terminate the current CUDA kernel through CUDA's device trap builtin.
[[noreturn]] __device__ __forceinline__ void trap() { __trap(); }

/// Terminate the current CUDA kernel when a device-side invariant fails.
/// \param condition True when the current CUDA thread should trap.
__device__ __forceinline__ void trap_if(bool condition) {
  if (condition) {
    trap();
  }
}

} // namespace xpool::utils::device
