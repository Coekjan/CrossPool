#pragma once

/// \file xpool/fabric/atnagent.cuh
/// \brief Attention-side execution entry for one Transport request.

#include <xpool/ffn.hpp>
#include <xpool/fabric/arena.cuh>
#include <xpool/macros.hpp>
#include <xpool/transport/arena.cuh>

namespace xpool::fabric::atnagent {

/// Execute one AtnAgent request through the joined Fabric generation.
/// \param arena Joined process-local Fabric arena view.
/// \param transport_arena Source endpoint, mailbox, payloads, and optional DP counts.
/// \return Result for the Transport Resident to publish into its local mailbox.
/// \pre transport_arena contains a Transport-validated immutable request.
XPOOL_DEVICE_FN xpool::ffn::ResultCode execute(const xpool::fabric::ArenaView &arena,
                                                  const xpool::transport::ArenaView &transport_arena);

} // namespace xpool::fabric::atnagent
