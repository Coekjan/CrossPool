#pragma once

/// \file xpool/ffnagent/projection.hpp
/// \brief One-time native FFN execution installation input.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <c10/core/ScalarType.h>

namespace xpool::ffnagent {

/// Canonical local Dense weight addresses used during capture or service.
struct DenseBindingResourceProjection {
  /// Device address of the fused gate/up projection weights.
  std::uintptr_t gate_up_weight_address;
  /// Device address of the down projection weights.
  std::uintptr_t down_weight_address;

  constexpr bool operator==(const DenseBindingResourceProjection &other) const = default;
};

/// Router-owner-only weight addresses used during capture or service.
struct MoeRouterBindingResourceProjection {
  /// Device address of router weights on the router-owning FfnAgent.
  std::uintptr_t weight_address;
  /// Optional device address of the model-specific router correction bias.
  std::optional<std::uintptr_t> correction_bias_address;

  constexpr bool operator==(const MoeRouterBindingResourceProjection &other) const = default;
};

/// Canonical local MoE weight addresses used during capture or service.
struct MoeBindingResourceProjection {
  /// Device address of local expert gate/up weights.
  std::uintptr_t expert_gate_up_weight_address;
  /// Device address of local expert down weights.
  std::uintptr_t expert_down_weight_address;
  /// Router resources when this FfnAgent owns routing for the layer.
  std::optional<MoeRouterBindingResourceProjection> router;

  constexpr bool operator==(const MoeBindingResourceProjection &other) const = default;
};

/// Canonical Dense or MoE weight-address representation.
using BindingResourceProjection =
    std::variant<DenseBindingResourceProjection, MoeBindingResourceProjection>;

/// One captured gated-Dense computation body and its Primary/Control captures.
struct DenseExecutionSignatureProjection {
  /// Dtype used by Fabric payloads and graph input/output tensors.
  c10::ScalarType payload_dtype;
  /// Fixed row Capacity captured by this signature.
  std::size_t payload_row_capacity;
  /// Model hidden width.
  std::size_t hidden_size;
  /// Tensor-parallel local intermediate width.
  std::size_t local_intermediate_size;
  /// Address of the Primary capture graph.
  std::uintptr_t primary_graph_address;
  /// Address of the Control capture graph used for rebasing discovery.
  std::uintptr_t control_graph_address;
  /// Device address used as graph input during capture.
  std::uintptr_t capture_input_address;
  /// Device address used as graph partial output during capture.
  std::uintptr_t capture_partial_address;
  /// Device address of the workspace used during capture.
  std::uintptr_t capture_workspace_address;
  /// Exact logical byte extent used to qualify and rebase captured workspace pointers.
  std::size_t compute_workspace_bytes;
  /// Canonical resources used by the Primary capture.
  DenseBindingResourceProjection primary_capture_resources;
  /// Canonical resources used by the Control capture.
  DenseBindingResourceProjection control_capture_resources;

  constexpr bool operator==(const DenseExecutionSignatureProjection &other) const = default;
};

/// One captured MoE computation body and its Primary/Control captures.
struct MoeExecutionSignatureProjection {
  /// Dtype used by Fabric payloads and graph input/output tensors.
  c10::ScalarType payload_dtype;
  /// Fixed row Capacity captured by this signature.
  std::size_t payload_row_capacity;
  /// Model hidden width.
  std::size_t hidden_size;
  /// Tensor-parallel local intermediate width per expert.
  std::size_t local_intermediate_size;
  /// Total expert count represented by the captured routing operation.
  std::size_t expert_count;
  /// Number of experts selected per row.
  std::size_t effective_topk;
  /// Optional routed-expert subset count used by model-specific routing.
  std::optional<std::size_t> routed_expert_count;
  /// Address of the Primary capture graph.
  std::uintptr_t primary_graph_address;
  /// Address of the Control capture graph used for rebasing discovery.
  std::uintptr_t control_graph_address;
  /// Device address used as graph input during capture.
  std::uintptr_t capture_input_address;
  /// Device address used as graph partial output during capture.
  std::uintptr_t capture_partial_address;
  /// Device address of the workspace used during capture.
  std::uintptr_t capture_workspace_address;
  /// Exact logical byte extent used to qualify and rebase captured workspace pointers.
  std::size_t compute_workspace_bytes;
  /// Device address used for routing metadata during capture.
  std::uintptr_t capture_routing_metadata_address;
  /// Optional device address used for dynamic live-row count during capture.
  std::optional<std::uintptr_t> capture_payload_rows_address;
  /// Canonical resources used by the Primary capture.
  MoeBindingResourceProjection primary_capture_resources;
  /// Canonical resources used by the Control capture.
  MoeBindingResourceProjection control_capture_resources;

  constexpr bool operator==(const MoeExecutionSignatureProjection &other) const = default;
};

/// Captured Dense or MoE signature accepted by native installation.
using ExecutionSignatureProjection = std::variant<DenseExecutionSignatureProjection, MoeExecutionSignatureProjection>;

/// One Plan-addressed local layer and its increasing-Capacity signatures.
struct LayerExecutionProjection {
  /// Fabric instance index addressed by this layer.
  std::size_t instance_index;
  /// Model-local layer ordinal.
  std::size_t layer_ordinal;
  /// Increasing-Capacity signature indices available to this layer.
  std::vector<std::size_t> execution_signature_indices;
  /// Layer-specific weight addresses installed into the shared signatures.
  BindingResourceProjection layer_resource_targets;

  bool operator==(const LayerExecutionProjection &other) const = default;
};

/// Complete one-time installation input for one FfnAgent PE.
struct ExecutionProjection {
  /// Unique local execution signatures in deterministic installation order.
  std::vector<ExecutionSignatureProjection> signatures;
  /// Locally placed layers in deterministic first-use order.
  std::vector<LayerExecutionProjection> layers;

  /// Validate intrinsic structure and deterministic first-use ordering.
  /// \throws c10::Error when the projection is malformed.
  void validate() const;

  bool operator==(const ExecutionProjection &other) const = default;
};

} // namespace xpool::ffnagent
