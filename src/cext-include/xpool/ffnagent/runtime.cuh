#pragma once

/// \file xpool/ffnagent/runtime.cuh
/// \brief Private device tables used by installed FfnAgent Lane Graphs.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

#include <xpool/fabric/protocol.hpp>

namespace xpool::ffnagent {

/// One device-updatable eight-byte kernel parameter site.
struct BindingSite {
  /// Device-updatable CUDA Graph node containing the parameter.
  cudaGraphDeviceNode_t node;
  /// Byte offset of the parameter within the node's packed argument buffer.
  std::size_t parameter_offset_bytes;
  /// Byte offset of the replacement address within LayerBindingValues.
  std::size_t value_offset_bytes;
};

/// Complete layer-resource address assignment shared across lanes.
struct LayerBindingValues {
  /// Device address of Dense or expert gate/up weights.
  std::uintptr_t gate_up_weight_address;
  /// Device address of Dense or expert down weights.
  std::uintptr_t down_weight_address;
  /// Device address of router weights, or zero for non-owning layers.
  std::uintptr_t router_weight_address;
  /// Device address of router correction bias, or zero when absent.
  std::uintptr_t router_correction_bias_address;
};

/// One lane/signature slice in the lane-owned Binding Site table.
struct DiscoveredBindingSchema {
  /// First BindingSite in the lane-owned flat table.
  std::size_t site_begin;
  /// Number of BindingSites belonging to this signature.
  std::size_t site_count;
};

/// One admitted Capacity and its lane Compute branch ordinal.
struct CapacityExecutionEntry {
  /// Fixed row Capacity admitted by this entry.
  std::size_t payload_row_capacity;
  /// Index of the execution signature captured for this Capacity.
  std::size_t execution_signature_index;
};

/// One exact local layer and its immutable bindings and Capacity rows.
struct LayerExecutionEntry {
  /// Fabric instance index addressed by the layer.
  std::size_t instance_index;
  /// Model-local layer ordinal.
  std::size_t layer_ordinal;
  /// First entry in the shared Capacity table.
  std::size_t capacity_begin;
  /// Number of increasing-Capacity entries owned by the layer.
  std::size_t capacity_count;
  /// Layer-specific addresses written into device-updatable graph nodes.
  LayerBindingValues binding_values;
  /// Whether execution must wait for routing metadata.
  bool requires_routing_metadata;
};

/// Mutable state owned by exactly one long-running Lane Graph.
struct LaneRuntimeState {
  /// Current lane execution copied after lease acquisition.
  xpool::fabric::LaneExecution execution;
  /// Most recently completed lease sequence.
  std::uint64_t last_lease_sequence;
  /// Capacity selected for the current execution.
  std::size_t selected_payload_row_capacity;
  /// Signature selected for the current execution.
  std::size_t selected_execution_signature_index;
  /// Local layer table index selected for the current execution.
  std::size_t selected_layer_entry_index;
  /// This FfnAgent's tensor-parallel rank for the selected layer.
  std::size_t local_tp_rank;
  /// GPU global-timer nanoseconds captured before waiting for peer partials;
  /// zero means the current execution has not started that wait.
  std::uint64_t peer_partial_wait_started_at;
  /// Result code produced by the current execution.
  xpool::ffn::ResultCode result_code;
  /// Whether the resident lane graph must exit after the current iteration.
  bool terminal;
};

static_assert(sizeof(std::uintptr_t) == 8);
static_assert(std::is_trivially_copyable_v<BindingSite>);
static_assert(std::is_standard_layout_v<LayerBindingValues>);
static_assert(std::is_trivially_copyable_v<DiscoveredBindingSchema>);
static_assert(std::is_trivially_copyable_v<CapacityExecutionEntry>);
static_assert(std::is_trivially_copyable_v<LayerExecutionEntry>);
static_assert(std::is_trivially_copyable_v<LaneRuntimeState>);

} // namespace xpool::ffnagent
