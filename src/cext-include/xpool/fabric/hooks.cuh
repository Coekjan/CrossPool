#pragma once

/// \file xpool/fabric/hooks.cuh
/// \brief Device extension points owned by Fabric execution.

#include <cstddef>
#include <cstdint>

#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/hooks.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/hooks/registry.hpp>

namespace xpool::hooks {

/// Device context for one AtnAgent-side Fabric protocol observation.
struct FabricAtnAgentProtocolEvent::Context {
  /// Process-local Fabric arena.
  xpool::fabric::ArenaView arena;
  /// Fabric instance addressed by the event.
  std::size_t instance_index;
  /// Protocol transition being observed.
  Kind kind;
  /// Submission payload for SubmissionPrepared; otherwise null.
  const xpool::fabric::Submission *submission = nullptr;
  /// Admission payload for AdmissionObserved; otherwise null.
  const xpool::fabric::Admission *admission = nullptr;
  /// Commit payload for OutputCommitObserved; otherwise null.
  const xpool::fabric::OutputCommit *commit = nullptr;
};

/// Device context for one Coordinator-side Fabric protocol observation.
struct FabricCoordinatorProtocolEvent::Context {
  /// Process-local Fabric arena.
  xpool::fabric::ArenaView arena;
  /// Fabric instance addressed by the event.
  std::size_t instance_index;
  /// Protocol transition being observed.
  Kind kind;
  /// Invocation payload for Enqueued; otherwise null.
  const xpool::fabric::Invocation *invocation = nullptr;
  /// FIFO scheduler ticket for Enqueued.
  std::uint64_t ready_ticket = 0;
  /// Selected lane for Scheduled.
  std::size_t executor_lane_index = 0;
  /// Selected lease sequence for Scheduled.
  std::uint64_t executor_lease_sequence = 0;
};

/// Device context for one FfnAgent-side Fabric protocol observation.
struct FabricFfnAgentProtocolEvent::Context {
  /// Process-local Fabric arena.
  xpool::fabric::ArenaView arena;
  /// Protocol transition being observed.
  Kind kind;
  /// Active Lane execution payload supplied for every FfnAgent event.
  const xpool::fabric::LaneExecution *execution = nullptr;
  /// Process-local executor lane index.
  std::size_t executor_lane_index = 0;
  /// Installed graph Capacity for LaneExecutionObserved or RoutingMetadataPublished.
  std::size_t payload_row_capacity = 0;
  /// Delivery variant for LaneExecutionObserved.
  xpool::fabric::DeliveryVariant delivery = xpool::fabric::DeliveryVariant::DirectPartial;
};

} // namespace xpool::hooks
