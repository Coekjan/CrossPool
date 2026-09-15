#pragma once

/// \file xpool/ffnagent/parameterization.hpp
/// \brief Internal CUDA Graph parameterization seam.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <cuda/std/span>
#include <cuda_runtime.h>

#include <xpool/ffnagent/runtime.cuh>

namespace xpool::ffnagent {

/// One weight-address delta used to discover and rewrite parameter sites.
struct ResourceReplacement {
  /// Address observed in the Primary Graph.
  std::uintptr_t primary_address;
  /// Corresponding address observed in the Control Graph.
  std::uintptr_t control_address;
  /// Lane-independent initial service address.
  std::uintptr_t target_address;
  /// Byte offset of this address within LayerBindingValues.
  std::size_t value_offset_bytes;
};

/// One captured storage range rebased into lane-owned storage.
struct LaneAddressReplacement {
  /// Beginning of the captured source range.
  std::uintptr_t capture_address;
  /// Byte extent of the captured source range.
  std::size_t bytes;
  /// Beginning of the lane-owned target range.
  std::uintptr_t target_address;
};

/// One node together with the graph that directly owns it.
struct PrimaryGraphLocation {
  /// Direct owner of node.
  cudaGraph_t graph;
  /// Located Primary Graph node.
  cudaGraphNode_t node;
};

/// Retained metadata produced while parameterizing one Primary Graph clone.
struct GraphParameterization {
  /// Device-updatable weight-address sites used during service.
  std::vector<BindingSite> binding_sites;
  /// Primary Graph node after which routing publication must be inserted.
  std::optional<PrimaryGraphLocation> routing_finalization;
};

/// Parameterize one lane-owned Primary Graph clone using its Control Graph.
/// \param primary_graph Graph clone rewritten in place for one Executor Lane.
/// \param control_graph Read-only topology-identical capture used to discover
/// weight parameter sites.
/// \param resources Required Primary/Control weight-address differences and
/// their service targets.
/// \param lane_replacements Required capture-address ranges and lane targets.
/// \param capture_routing_metadata_address Captured beginning of packed routing
/// metadata, or zero for Dense execution.
/// \param capture_routing_weights_address Captured beginning of routing weights,
/// or zero for Dense execution.
/// \param capture_payload_rows_address Captured live-row scalar, or zero when
/// routing is produced by another TP rank.
/// \return Service metadata retained after both source Graph owners are released.
/// \throws c10::Error when Graph schemas differ, a parameter delta is undeclared,
/// or a required resource or lane replacement is absent.
GraphParameterization parameterize_graph(cudaGraph_t primary_graph, cudaGraph_t control_graph,
                                         cuda::std::span<const ResourceReplacement> resources,
                                         cuda::std::span<const LaneAddressReplacement> lane_replacements,
                                         std::uintptr_t capture_routing_metadata_address,
                                         std::uintptr_t capture_routing_weights_address,
                                         std::uintptr_t capture_payload_rows_address);

} // namespace xpool::ffnagent
