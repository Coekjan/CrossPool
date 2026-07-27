#pragma once

#include <cuda_runtime_api.h>

#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/layout.hpp>

namespace xpool::fabric {

/// Launch shutdown-aware Fabric progress for one FfnAgent PE.
/// \param arena Joined process-local symmetric arena view.
/// \param layout Host-owned validated arena geometry.
/// \param stream Nonblocking stream that owns the resident kernel.
void launch_ffnagent_kernel(FabricArenaView arena, const FabricArenaLayout &layout, cudaStream_t stream);

} // namespace xpool::fabric
