#pragma once

/// \file xpool/transport/hooks.cuh
/// \brief Device extension points owned by Transport execution.

#include <cstddef>
#include <cstdint>

#include <xpool/ffn.hpp>
#include <xpool/hooks/registry.hpp>
#include <xpool/transport/arena.hpp>
#include <xpool/transport/hooks.hpp>

namespace xpool::hooks {

/// Device context for one Instance-side Transport protocol observation.
struct TransportInstanceProtocolEvent::Context {
  /// Process-local Transport arena.
  xpool::transport::ArenaView arena;
  /// Protocol transition being observed.
  Kind kind;
  /// Live payload rows for row-dependent transitions.
  std::size_t payload_rows = 0;
  /// Request metadata for RequestStagingStarted; otherwise null.
  const xpool::transport::RequestMetadata *request = nullptr;
  /// Result code for ResultObserved and Closed; otherwise the sentinel default.
  xpool::ffn::ResultCode result_code{xpool::ffn::ResultCode::ProtocolMismatch};
};

/// Device context for one AtnAgent-side Transport protocol observation.
struct TransportAtnAgentProtocolEvent::Context {
  /// Process-local Transport arena.
  xpool::transport::ArenaView arena;
  /// Protocol transition being observed.
  Kind kind;
  /// Live payload rows for row-dependent transitions.
  std::size_t payload_rows = 0;
  /// Request metadata for RequestObserved; otherwise null.
  const xpool::transport::RequestMetadata *request = nullptr;
  /// Result code for ResultPublished; otherwise the sentinel default.
  xpool::ffn::ResultCode result_code{xpool::ffn::ResultCode::ProtocolMismatch};
};

} // namespace xpool::hooks
