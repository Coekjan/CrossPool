#pragma once

/// \file xpool/fabric/coordinator.hpp
/// \brief Host launch boundary for the generation-scoped Fabric Coordinator.

#include <cuda_runtime_api.h>

#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/ffnagent.hpp>
#include <xpool/fabric/layout.hpp>

namespace xpool::fabric {

/// Launch the Fabric Coordinator.
/// \pre arena, control activation count, control Coordinator Scheduler, and stream are non-null.
/// \post Coordinator progress is enqueued asynchronously on stream.
/// \throws c10::Error when the cooperative launch cannot be configured or enqueued.
void launch_coordinator(ArenaView arena, const ArenaLayout &layout, FfnAgentControlView control, cudaStream_t stream);

} // namespace xpool::fabric
