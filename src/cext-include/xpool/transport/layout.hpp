#pragma once

/// \file xpool/transport/layout.hpp
/// \brief Immutable geometry of one rank-local CUDA IPC transport arena.

#include <cstddef>
#include <cstdint>

#include <c10/core/ScalarType.h>

#include <xpool/arena.hpp>

namespace xpool::transport {

/// Stable magic identifying a Transport arena layout header.
inline constexpr std::uint64_t kTransportArenaMagic = 0x3141544c4f4f5058ULL;

/// Immutable identity, topology, geometry, and region offsets for one arena.
struct ArenaLayout {
  /// Common ABI, size, and mutable-state offset header at arena offset zero.
  xpool::arena::LayoutHeader header;
  /// Configuration-order Instance index used by the Fabric Instance table.
  std::size_t instance_index;
  /// Instance-rank identity within this model topology.
  std::size_t instance_rank;
  /// Attention tensor-parallel coordinate.
  std::size_t atn_tp_rank;
  /// Attention tensor-parallel width.
  std::size_t atn_tp_size;
  /// Attention data-parallel coordinate.
  std::size_t atn_dp_rank;
  /// Attention data-parallel width.
  std::size_t atn_dp_size;
  /// Maximum physical hidden-state rows accepted by one request.
  std::size_t payload_row_capacity;
  /// Hidden-state columns in every request.
  std::size_t hidden_size;
  /// Bytes occupied by one physical payload row.
  std::size_t payload_row_bytes;
  /// Hidden-state element type for both payloads.
  c10::ScalarType payload_dtype;
  /// Offset of the sole Mailbox.
  std::size_t mailbox_offset;
  /// Offset of the fixed input payload byte range.
  std::size_t input_payload_offset;
  /// Offset of the fixed output payload byte range.
  std::size_t output_payload_offset;
  /// Offset of physical rows per DP rank, or zero when atn_dp_size is one.
  std::size_t dp_rank_payload_rows_offset;
  /// Plan one complete aligned Transport arena.
  /// \throws c10::Error when geometry is invalid or size arithmetic overflows.
  static ArenaLayout create(std::size_t instance_index, std::size_t instance_rank, std::size_t atn_tp_rank,
                            std::size_t atn_tp_size, std::size_t atn_dp_rank, std::size_t atn_dp_size,
                            std::size_t payload_row_capacity, std::size_t hidden_size, c10::ScalarType payload_dtype);
  /// Validate every identity, topology, geometry, offset, and total-size fact.
  /// \throws c10::Error when any stored layout fact is inconsistent.
  void validate() const;
  constexpr bool operator==(const ArenaLayout &) const = default;
};

} // namespace xpool::transport
