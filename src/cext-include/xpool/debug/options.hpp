#pragma once

/// \file xpool/debug/options.hpp
/// \brief Host-side native debug option state.

#include <cstdint>

#include <xpool/abi.hpp>

namespace xpool::debug {

/// Install process-wide native debug options for one CUDA device.
/// \param cuda_device CUDA device whose device-side debug state should be
/// updated.
/// \param debug_options_mask Raw debug option mask resolved from xpool config.
/// \throws c10::Error if CUDA state update fails.
void init(std::int64_t cuda_device, std::int64_t debug_options_mask);

/// Return a snapshot of the host-side native debug options.
/// \return Process-wide debug options last installed by init().
xpool::abi::DebugOptions options();

} // namespace xpool::debug
