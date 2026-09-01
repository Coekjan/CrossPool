#pragma once

/// \file xpool/fabric/projection.hpp
/// \brief Semantic input for one generation-scoped NVSHMEM Fabric.

#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <c10/core/ScalarType.h>

#include <xpool/ffn.hpp>
#include <xpool/fabric/layout.hpp>
#include <xpool/fabric/scheduler.hpp>
#include <xpool/utils/hex.hpp>

namespace xpool::fabric {

/// Opaque NVSHMEM unique id with canonical hexadecimal projection.
using Uid = xpool::utils::hex::HexValue<nvshmemx_uniqueid_t>;

/// One ordered FFN layer supplied to native Fabric join.
struct InstanceLayerProjection {
  /// Model decoder layer identifier used for executor binding.
  std::size_t layer_id;
  /// Valid structural executor kind for this layer.
  xpool::ffn::LayerKind kind;
  /// Final routing width; zero for Dense layers.
  std::size_t effective_topk;
  /// Ordered local FfnAgent indices, one per FFN TP rank.
  std::vector<std::size_t> ffnagent_indices;

  bool operator==(const InstanceLayerProjection &) const = default;
};

/// One Instance's topology, capacities, and ordered FFN layers at native join.
struct InstanceProjection {
  /// Maximum Decode rows admitted by the unified Lane payload storage.
  std::size_t decode_payload_row_capacity;
  /// Maximum Prefill rows admitted by the unified Lane payload storage.
  std::size_t prefill_payload_row_capacity;
  /// Hidden-state element type shared by this model's Fabric payloads.
  c10::ScalarType payload_dtype;
  /// Hidden-state columns in every invocation for this model.
  std::size_t hidden_size;
  /// Whether Group-Sum Complete output is admitted for this Instance.
  bool group_sum_complete_admitted;
  /// Tensor-parallel AtnAgent width for this model.
  std::size_t atn_tp_size;
  /// Data-parallel AtnAgent width for this model.
  std::size_t atn_dp_size;
  /// Ordered AtnAgent indices in TP-fastest rank order.
  std::vector<std::size_t> atnagent_indices;
  /// Decoder FFN layers in canonical execution order.
  std::vector<InstanceLayerProjection> layers;

  bool operator==(const InstanceProjection &) const = default;
};

/// Complete semantic input for one process joining an NVSHMEM Fabric world.
struct ArenaProjection {
  /// High word of the nonzero daemon-authored Generation identity.
  std::uint64_t generation_high;
  /// Low word of the nonzero daemon-authored Generation identity.
  std::uint64_t generation_low;
  /// Opaque NVSHMEM bootstrap UID shared by every participant.
  Uid uid;
  /// Number of AtnAgent PEs in the canonical prefix.
  std::size_t atnagent_count;
  /// Number of FfnAgent PEs in the canonical suffix.
  std::size_t ffnagent_count;
  /// Number of independently admitted distributed FFN Executors.
  std::size_t executor_lane_count;
  /// Immutable policy used to construct the arena Scheduler.
  SchedulerPolicy scheduler;
  /// Instances in configuration order.
  std::vector<InstanceProjection> instances;

  /// Return the total participant count.
  /// \return Sum of AtnAgent and FfnAgent PE counts.
  std::size_t pe_count() const { return atnagent_count + ffnagent_count; }

  /// Return the first FfnAgent PE and sole Coordinator owner.
  /// \return NVSHMEM PE index of the Coordinator.
  int coordinator_pe() const { return static_cast<int>(atnagent_count); }

  /// Validate UID, topology, capacities, policy, and ordered layers.
  /// \throws c10::Error when topology, geometry, membership, or ordering is inconsistent.
  void validate() const;

  bool operator==(const ArenaProjection &) const = default;
};

} // namespace xpool::fabric
