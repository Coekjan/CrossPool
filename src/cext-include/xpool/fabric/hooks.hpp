#pragma once

/// \file xpool/fabric/hooks.hpp
/// \brief Host and host-visible Device extension points owned by Fabric.

#include <c10/core/Device.h>

#include <cstdint>

#include <xpool/fabric/layout.hpp>
#include <xpool/fabric/projection.hpp>
#include <xpool/hooks/registry.hpp>

#if defined(__CUDACC__)
#include <xpool/hooks/registry.cuh>
#endif

namespace xpool::hooks {

/// Observes a process after it joins and materializes Fabric.
struct FabricJoinPostEvent final : HostObservePoint {
  /// Joined Fabric metadata exposed to Host observers.
  struct Context {
    /// CUDA device that owns the Fabric participant.
    c10::DeviceIndex cuda_device;
    /// NVSHMEM PE assigned to the participant.
    int pe;
    /// Materialized symmetric-arena layout.
    const xpool::fabric::ArenaLayout &layout;
    /// Joined generation topology.
    const xpool::fabric::ArenaProjection &projection;
  };

  /// Dispatch all registered Host observers.
  XPOOL_HOST_HOOK_POINT(FabricJoinPostEvent);
};

/// Observes a Fabric participant before native finalization.
struct FabricFinalizePreEvent final : HostObservePoint {
  /// Finalizing participant metadata exposed to Host observers.
  struct Context {
    /// CUDA device that owns the Fabric participant.
    c10::DeviceIndex cuda_device;
  };

  /// Dispatch all registered Host observers.
  XPOOL_HOST_HOOK_POINT(FabricFinalizePreEvent);
};

/// Observes AtnAgent-side Fabric protocol transitions on the device.
struct FabricAtnAgentProtocolEvent final : DeviceObservePoint {
  /// Ordered AtnAgent protocol transitions.
  enum class Kind : std::uint32_t {
    SubmissionPrepared,
    SubmissionPublished,
    AdmissionObserved,
    InputReadyPublished,
    OutputCommitObserved,
    OutputAcknowledgementPublished,
    Count,
  };

  struct Context;

#if defined(__CUDACC__)
  /// Dispatch all registered Device observers.
  XPOOL_DEVICE_HOOK_POINT(FabricAtnAgentProtocolEvent);
#endif
};

/// Observes Coordinator-side Fabric protocol transitions on the device.
struct FabricCoordinatorProtocolEvent final : DeviceObservePoint {
  /// Ordered Coordinator protocol transitions.
  enum class Kind : std::uint32_t {
    Enqueued,
    Scheduled,
    AdmissionPublished,
    LaneExecutionPublished,
    FfnAgentCompletionsObserved,
    OutputCommitPublished,
    OutputAcknowledgementsObserved,
    LaneReleased,
    Count,
  };

  struct Context;

#if defined(__CUDACC__)
  /// Dispatch all registered Device observers.
  XPOOL_DEVICE_HOOK_POINT(FabricCoordinatorProtocolEvent);
#endif
};

/// Observes FfnAgent-side Fabric protocol transitions on the device.
struct FabricFfnAgentProtocolEvent final : DeviceObservePoint {
  /// Ordered FfnAgent protocol transitions.
  enum class Kind : std::uint32_t {
    LaneExecutionObserved,
    InputReadyObserved,
    RoutingMetadataPublished,
    RoutingMetadataObserved,
    ComputeStarted,
    ComputeCompleted,
    PartialReadyPublished,
    PeerPartialsReadyObserved,
    CompletionPublished,
    Count,
  };

  struct Context;

#if defined(__CUDACC__)
  /// Dispatch all registered Device observers.
  XPOOL_DEVICE_HOOK_POINT(FabricFfnAgentProtocolEvent);
#endif
};

} // namespace xpool::hooks
