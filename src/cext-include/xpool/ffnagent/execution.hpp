#pragma once

/// \file xpool/ffnagent/execution.hpp
/// \brief One-time native FFN execution installation input.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <c10/core/ScalarType.h>

namespace xpool::ffnagent {

/// Encoded values, optional scales and optional additive bias of one projection.
struct ProjectionBindingView {
  /// Nonzero device address of the retained values Tensor.
  std::uintptr_t values;
  /// Positive logical byte extent of the values Tensor.
  std::size_t values_bytes;
  /// Device address of scale storage, when present.
  std::optional<std::uintptr_t> scales;
  /// Scale Tensor bytes, or zero when scales are absent.
  std::size_t scales_bytes;
  /// Device address of additive projection bias, when present.
  std::optional<std::uintptr_t> bias;
  /// Bias Tensor bytes, or zero when bias is absent.
  std::size_t bias_bytes;

  /// Check resource presence, nonnull addresses and corresponding byte extents.
  /// \throws c10::Error when an address/extent pair is malformed.
  void validate() const;

  /// Compare resource presence and byte extents, ignoring addresses and encoding.
  bool schema_matches(const ProjectionBindingView &other) const;

  constexpr bool operator==(const ProjectionBindingView &other) const = default;
};

/// Canonical local Dense resources used during capture or service.
struct DenseBindingView {
  /// Fused gate/up projection resources.
  ProjectionBindingView gate_up;
  /// Down projection resources.
  ProjectionBindingView down;

  constexpr bool operator==(const DenseBindingView &other) const = default;
};

/// Router-owner-only weight addresses used during capture or service.
struct MoeRouterBindingView {
  /// Projection resources on the router-owning FfnAgent.
  ProjectionBindingView projection;
  /// Optional device address of the model-specific router correction bias.
  std::optional<std::uintptr_t> correction_bias_address;
  /// Correction-bias Tensor bytes, or zero when absent.
  std::size_t correction_bias_bytes;

  /// Check projection resources and correction-bias presence/extent.
  /// \throws c10::Error when a resource is malformed.
  void validate() const;

  /// Compare resource presence and byte extents without comparing addresses.
  bool schema_matches(const MoeRouterBindingView &other) const;

  constexpr bool operator==(const MoeRouterBindingView &other) const = default;
};

/// Canonical local MoE weight addresses used during capture or service.
struct MoeBindingView {
  /// Local Expert gate/up projection resources.
  ProjectionBindingView gate_up;
  /// Local Expert down projection resources.
  ProjectionBindingView down;
  /// Router resources when this FfnAgent owns routing for the layer.
  std::optional<MoeRouterBindingView> router;

  constexpr bool operator==(const MoeBindingView &other) const = default;
};

/// Canonical Dense or MoE weight-address representation.
using BindingView = std::variant<DenseBindingView, MoeBindingView>;

/// One captured gated-Dense computation body and its Primary/Control captures.
struct DenseCaptureDescriptor {
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
  DenseBindingView primary_capture_resources;
  /// Canonical resources used by the Control capture.
  DenseBindingView control_capture_resources;

  constexpr bool operator==(const DenseCaptureDescriptor &other) const = default;
};

/// One captured MoE computation body and its Primary/Control captures.
struct MoeCaptureDescriptor {
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
  MoeBindingView primary_capture_resources;
  /// Canonical resources used by the Control capture.
  MoeBindingView control_capture_resources;

  constexpr bool operator==(const MoeCaptureDescriptor &other) const = default;
};

/// Captured Dense or MoE signature accepted by native installation.
using CaptureDescriptor = std::variant<DenseCaptureDescriptor, MoeCaptureDescriptor>;

/// One Plan-addressed local layer and its increasing-Capacity signatures.
struct LayerExecutionDescriptor {
  /// Fabric instance index addressed by this layer.
  std::size_t instance_index;
  /// Model-local layer ordinal.
  std::size_t layer_ordinal;
  /// Increasing-Capacity signature indices available to this layer.
  std::vector<std::size_t> execution_signature_indices;
  /// Layer-specific weight addresses installed into the shared signatures.
  BindingView layer_resource_targets;

  bool operator==(const LayerExecutionDescriptor &other) const = default;
};

/// Complete one-time installation input for one FfnAgent PE.
struct ExecutionInstallDescriptor {
  /// Unique local execution signatures in deterministic installation order.
  std::vector<CaptureDescriptor> signatures;
  /// Locally placed layers in deterministic first-use order.
  std::vector<LayerExecutionDescriptor> layers;

  /// Validate intrinsic structure and deterministic first-use ordering.
  /// \throws c10::Error when the projection is malformed.
  void validate() const;

  bool operator==(const ExecutionInstallDescriptor &other) const = default;
};

} // namespace xpool::ffnagent
