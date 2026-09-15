#pragma once

/// \file xpool/utils/time.cuh
/// \brief CUDA device clock helpers shared by polling and tracing.

#include <cstdint>

#include <cuda/ptx>

#include <xpool/macros.hpp>

namespace xpool::utils::time {

/// Read the current GPU global timer.
/// \return Device-global timestamp in nanoseconds.
/// \note Timestamp differences are meaningful only within one GPU clock
/// domain.
XPOOL_DEVICE_FN inline std::uint64_t now() { return cuda::ptx::get_sreg_globaltimer(); }

} // namespace xpool::utils::time
