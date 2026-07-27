#pragma once

/// \file xpool/abort.hpp
/// \brief Immediate host and CUDA device termination primitives.

#include <cstdlib>

#include <xpool/macros.hpp>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace xpool {

/// Terminate execution after an unrecoverable internal failure.
/// \post The host process aborts or the current CUDA kernel traps.
[[noreturn]] XPOOL_HOST_DEVICE_FN inline void abort() noexcept {
#if defined(__CUDA_ARCH__)
  __trap();
#else
  std::abort();
#endif
}

/// Terminate execution when an unrecoverable internal condition is true.
/// \param condition True when execution must terminate immediately.
XPOOL_HOST_DEVICE_FN inline void abort_if(bool condition) noexcept {
  if (condition) {
    xpool::abort();
  }
}

} // namespace xpool
