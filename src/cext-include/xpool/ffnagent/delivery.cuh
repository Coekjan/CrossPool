#pragma once

/// \file xpool/ffnagent/delivery.cuh
/// \brief Fixed-argument TP delivery entries installed in FfnAgent lane graphs.

#include <cstddef>

#include <xpool/fabric/arena.hpp>
#include <xpool/macros.hpp>

namespace xpool::ffnagent {

/// Stage one immutable rank-local Partial without TP reduction.
/// \param arena Stable process-local Fabric arena view.
/// \param executor_lane_index Stable lane whose current Record supplies live geometry.
XPOOL_KERNEL_FN void deliver_direct_partial_output(xpool::fabric::ArenaView arena,
                                                   std::size_t executor_lane_index);

/// Reduce one balanced row range in fixed PE order using FP32 accumulation.
/// The final sum is cast to the payload dtype once before remote staging; the
/// calling rank owns only the row range assigned by the delivery plan.
/// \param arena Stable process-local Fabric arena view.
/// \param executor_lane_index Stable lane whose current Record supplies live geometry.
XPOOL_KERNEL_FN void deliver_complete_output_range(xpool::fabric::ArenaView arena,
                                                   std::size_t executor_lane_index);

} // namespace xpool::ffnagent
