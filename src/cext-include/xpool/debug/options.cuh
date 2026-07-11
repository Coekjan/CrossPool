#pragma once

/// \file xpool/debug/options.cuh
/// \brief Device-side native debug option state.

#include <cstdint>

#include <xpool/abi.hpp>

namespace xpool::debug {

/// Device-side debug option bitmask installed by xpool::debug::init.
extern __device__ __constant__ xpool::abi::DebugOptions g_debug_options;

} // namespace xpool::debug
