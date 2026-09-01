#pragma once

/// \file xpool/ffnagent/hooks.hpp
/// \brief Host and CUDA Graph extension points owned by FfnAgent execution.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <xpool/fabric/projection.hpp>
#include <xpool/hooks/registry.hpp>

namespace xpool::hooks {

/// Observes the start of one native FFN execution installation.
struct FfnExecutionInstallPreEvent final : HostObservePoint {
  /// Installation metadata exposed to Host observers.
  struct Context {
    /// Joined Fabric topology for this generation.
    const xpool::fabric::ArenaProjection &fabric_projection;
    /// Fleet index of the installing FfnAgent.
    std::size_t ffnagent_index;
  };

  /// Dispatch all registered Host observers.
  XPOOL_HOST_HOOK_POINT(FfnExecutionInstallPreEvent);
};

/// Observes completion of native FFN execution finalization.
struct FfnExecutionFinalizePostEvent final : HostObservePoint {
  struct Context {};

  /// Dispatch all registered Host observers.
  XPOOL_HOST_HOOK_POINT(FfnExecutionFinalizePostEvent);
};

/// Observes one primary computation graph after native parameterization.
struct FfnPrimaryGraphParameterizePostEvent final : HostObservePoint {
  /// Parameterized graph metadata exposed to Host observers.
  struct Context {
    /// Index of the installed execution signature.
    std::size_t execution_signature_index;
    /// Parameterized primary graph.
    cudaGraph_t graph;
    /// Number of device-address binding sites discovered in the graph.
    std::size_t binding_site_count;
  };

  /// Dispatch all registered Host observers.
  XPOOL_HOST_HOOK_POINT(FfnPrimaryGraphParameterizePostEvent);
};

/// Observes one complete executor-lane graph after construction.
struct FfnLaneGraphBuildPostEvent final : HostObservePoint {
  /// Completed lane-graph metadata exposed to Host observers.
  struct Context {
    /// Process-local executor lane index.
    std::size_t executor_lane_index;
    /// Completed lane graph.
    cudaGraph_t graph;
    /// Conditional switch that selects an installed computation graph.
    cudaGraphNode_t compute_switch;
    /// Conditional switch that selects the delivery graph.
    cudaGraphNode_t delivery_switch;
  };

  /// Dispatch all registered Host observers.
  XPOOL_HOST_HOOK_POINT(FfnLaneGraphBuildPostEvent);
};

} // namespace xpool::hooks
