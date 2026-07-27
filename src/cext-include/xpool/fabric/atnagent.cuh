#pragma once

/// \file xpool/fabric/atnagent.cuh
/// \brief Attention-side execution entry for one Transport request.

#include <xpool/abi.hpp>
#include <xpool/fabric/arena.cuh>
#include <xpool/macros.hpp>
#include <xpool/transport/arena.cuh>
#include <xpool/transport/trace.cuh>

namespace xpool::fabric::atnagent {

/// Execute one AtnAgent request through the joined Fabric generation.
/// \param arena Joined process-local Fabric arena view.
/// \param transport_arena Source endpoint, mailbox, payloads, and optional DP counts.
/// \param transport_trace Optional Transport trace advanced at Fabric admission.
/// \return Result already published into the local Transport mailbox.
XPOOL_DEVICE_FN xpool::abi::FfnResultCode execute(
    const xpool::fabric::FabricArenaView &arena,
    const xpool::transport::TransportArenaView &transport_arena,
    xpool::transport::TransportTraceRecord *transport_trace);

} // namespace xpool::fabric::atnagent
