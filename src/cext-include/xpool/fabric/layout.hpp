#pragma once

/// \file xpool/fabric/layout.hpp
/// \brief Immutable geometry for one generation-scoped NVSHMEM Fabric arena.

#include <cstddef>
#include <cstdint>

#include <c10/core/ScalarType.h>

#include <xpool/abort.hpp>
#include <xpool/arena.hpp>
#include <xpool/ffn.hpp>
#include <xpool/macros.hpp>

namespace xpool::fabric {

/// Required alignment of each remotely signaled Fabric publication.
inline constexpr std::size_t kFabricPublicationAlignment = 256;

/// Fixed number of physical payload buffers owned by each executor lane.
inline constexpr std::size_t kExecutorLanePayloadBufferCount = 2;

/// Magic identifying a Fabric arena root at symmetric allocation offset zero.
inline constexpr std::uint64_t kFabricArenaMagic = 0x3141464c4f4f5058ULL;

/// One immutable Instance row in the Fabric topology table.
struct InstanceEntry {
  /// Hidden-state element type shared by this Instance's payloads.
  c10::ScalarType payload_dtype;
  /// Whether the generation admitted AtnAgent-side group-sum completion.
  bool group_sum_complete_admitted;

  /// Hidden-state width in payload elements.
  std::size_t hidden_size;
  /// Bytes occupied by one physical payload row.
  std::size_t payload_row_bytes;
  /// Maximum Decode rows admitted by the unified Lane payload storage.
  std::size_t decode_payload_row_capacity;
  /// Maximum Prefill rows admitted by the unified Lane payload storage.
  std::size_t prefill_payload_row_capacity;

  /// Tensor-parallel AtnAgent ranks per DP rank.
  std::size_t atn_tp_size;
  /// Data-parallel AtnAgent rank count.
  std::size_t atn_dp_size;

  /// First layer row in the flattened generation layer table.
  std::size_t layer_begin;
  /// Number of layer rows owned by this Instance.
  std::size_t layer_count;

  /// First PE entry in this Instance's flattened AtnAgent membership.
  std::size_t atnagent_pe_begin;
  /// First PE entry in this Instance's flattened FfnAgent membership.
  std::size_t ffnagent_pe_begin;
  /// FfnAgent tensor-parallel width used by every layer.
  std::size_t ffn_tp_size;

  /// Convert one Atn TP/DP coordinate to the Instance-local AtnAgent index.
  XPOOL_HOST_DEVICE_FN std::size_t atnagent_index(std::size_t atn_tp_rank, std::size_t atn_dp_rank) const {
    xpool::abort_if(atn_tp_rank >= atn_tp_size || atn_dp_rank >= atn_dp_size);
    return atn_dp_rank * atn_tp_size + atn_tp_rank;
  }

  /// Return the tensor-parallel coordinate of an Instance-local AtnAgent.
  XPOOL_HOST_DEVICE_FN std::size_t atn_tp_rank(std::size_t atnagent_index) const {
    xpool::abort_if(atn_tp_size == 0 || atnagent_index >= atn_tp_size * atn_dp_size);
    return atnagent_index % atn_tp_size;
  }

  /// Return the data-parallel coordinate of an Instance-local AtnAgent.
  XPOOL_HOST_DEVICE_FN std::size_t atn_dp_rank(std::size_t atnagent_index) const {
    xpool::abort_if(atn_tp_size == 0 || atnagent_index >= atn_tp_size * atn_dp_size);
    return atnagent_index / atn_tp_size;
  }

  constexpr bool operator==(const InstanceEntry &) const = default;
};

/// One immutable decoder-layer row in the flattened Fabric layer table.
struct LayerEntry {
  /// Model-defined layer identifier used to bind captured weights.
  std::size_t layer_id;
  /// Structural layer kind.
  xpool::ffn::LayerKind kind;
  /// Router-selected experts per row; zero for Dense layers.
  std::size_t effective_topk;

  constexpr bool operator==(const LayerEntry &) const = default;
};

/// Immutable root geometry stored at symmetric-arena offset zero.
struct ArenaLayout {
  /// Magic, ABI, and total allocation bytes at arena offset zero.
  xpool::arena::LayoutHeader header;

  /// Number of AtnAgent PEs in the generation.
  std::size_t atnagent_count;
  /// Number of FfnAgent PEs in the generation.
  std::size_t ffnagent_count;
  /// Number of configured Instance entries.
  std::size_t instance_count;
  /// Number of reusable Executor Lanes.
  std::size_t executor_lane_count;

  /// Number of rows in the flattened layer table.
  std::size_t layer_entry_count;
  /// Number of rows in the flattened AtnAgent PE table.
  std::size_t atnagent_pe_entry_count;
  /// Number of rows in the flattened FfnAgent PE table.
  std::size_t ffnagent_pe_entry_count;

  /// Byte offset of the immutable Instance table.
  std::size_t instance_entries_offset_bytes;
  /// Byte offset of the immutable layer table.
  std::size_t layer_entries_offset_bytes;
  /// Byte offset of the flattened AtnAgent PE table.
  std::size_t atnagent_pes_offset_bytes;
  /// Byte offset of the flattened FfnAgent PE table.
  std::size_t ffnagent_pes_offset_bytes;

  /// Byte offset of AtnAgent Submission publications.
  std::size_t submission_publications_offset_bytes;
  /// Byte offset of per-Instance Admission publications.
  std::size_t admission_publications_offset_bytes;
  /// Byte offset of per-Lane execution publications.
  std::size_t lane_execution_publications_offset_bytes;
  /// Byte offset of per-Lane input-ready publications.
  std::size_t input_ready_publications_offset_bytes;
  /// Byte offset of per-Lane routing-metadata publications.
  std::size_t routing_metadata_ready_publications_offset_bytes;
  /// Byte offset of per-FfnAgent, per-Lane partial-ready publications.
  std::size_t partial_ready_publications_offset_bytes;
  /// Byte offset of per-FfnAgent, per-Lane completion publications.
  std::size_t ffnagent_completion_publications_offset_bytes;
  /// Byte offset of per-Instance output commits.
  std::size_t output_commit_publications_offset_bytes;
  /// Byte offset of per-AtnAgent, per-Instance output acknowledgements.
  std::size_t output_acknowledgement_publications_offset_bytes;

  /// Byte offset of all Lane payload buffers.
  std::size_t lane_payload_storage_offset_bytes;
  /// Aligned byte capacity of one physical Lane payload buffer.
  std::size_t lane_payload_capacity_bytes;

  /// Byte offset of all Lane routing-metadata blocks.
  std::size_t routing_metadata_offset_bytes;
  /// Aligned byte stride of one Lane routing-metadata block.
  std::size_t routing_metadata_stride_bytes;

  /// Return the fixed Coordinator PE.
  /// The Coordinator is the first FfnAgent PE after the AtnAgent range.
  XPOOL_HOST_DEVICE_FN int coordinator_pe() const { return static_cast<int>(atnagent_count); }

  /// Materialize exact offsets and allocation size from semantic geometry.
  /// \throws c10::Error when topology/table counts or Lane payload capacity are
  /// zero, the PE count exceeds NVSHMEM's domain, or layout arithmetic overflows.
  static ArenaLayout create(std::size_t atnagent_count, std::size_t ffnagent_count, std::size_t instance_count,
                                  std::size_t executor_lane_count, std::size_t layer_entry_count,
                                  std::size_t atnagent_pe_entry_count, std::size_t ffnagent_pe_entry_count,
                                  std::size_t maximum_lane_payload_bytes,
                                  std::size_t maximum_routing_metadata_elements);

  /// Validate header, counts, offsets, alignment, and total allocation size.
  /// \throws c10::Error when the layout is internally inconsistent.
  void validate() const;

  constexpr bool operator==(const ArenaLayout &) const = default;
};

/// Return the exact native Fabric arena allocation size for semantic geometry.
/// \throws c10::Error under the same conditions as ArenaLayout::create.
std::size_t arena_allocation_bytes(std::size_t atnagent_count, std::size_t ffnagent_count, std::size_t instance_count,
                                   std::size_t executor_lane_count, std::size_t layer_entry_count,
                                   std::size_t atnagent_pe_entry_count, std::size_t ffnagent_pe_entry_count,
                                   std::size_t maximum_lane_payload_bytes,
                                   std::size_t maximum_routing_metadata_elements);

} // namespace xpool::fabric
