#pragma once

/// \file xpool/debug/options.cuh
/// \brief Device-side native debug option state.

#include <cstdint>

#include <xpool/debug/options.hpp>
#include <xpool/macros.hpp>

namespace xpool::debug {

/// Device-side typed debug options installed by xpool::debug::configure.
/// The symbol contains the all-disabled DebugOptions value before explicit
/// configuration and is accessed by options().
extern __device__ __constant__ DebugOptions options_d;

/// Return the device-side native debug options snapshot.
/// \return Explicitly configured options, or the all-disabled default before
/// configuration.
XPOOL_DEVICE_FN inline DebugOptions options() { return options_d; }

} // namespace xpool::debug
