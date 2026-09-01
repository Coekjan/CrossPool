#pragma once

/// \file xpool/debug/options.cuh
/// \brief Device-side native debug option state.

#include <cstdint>

#include <xpool/debug/options.hpp>
#include <xpool/macros.hpp>

namespace xpool::debug {

/// Device-side typed debug options installed by xpool::debug::configure.
/// The symbol contains the all-disabled Options value before explicit
/// configuration and is accessed by options().
extern XPOOL_DEVICE_CONST Options options_d;

/// Return the process-global native debug options.
/// \return Explicitly configured options, or the all-disabled default before
/// configuration.
XPOOL_HOST_DEVICE_FN inline const Options &options() {
#if defined(__CUDA_ARCH__)
  return options_d;
#else
  return options_h;
#endif
}

} // namespace xpool::debug
