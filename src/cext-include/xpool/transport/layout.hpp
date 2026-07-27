#pragma once

/// \file xpool/transport/layout.hpp
/// \brief Immutable geometry of one rank-local CUDA IPC transport arena.

#include <cstddef>
#include <cstdint>

#include <xpool/abi.hpp>
#include <xpool/arena.hpp>
#include <xpool/trace.hpp>

namespace xpool::transport {

/// Stable magic identifying a Transport arena layout header.
inline constexpr std::uint64_t kTransportArenaMagic = 0x3141544c4f4f5058ULL;

/// Immutable identity, topology, geometry, and region offsets for one arena.
struct TransportArenaLayout {
  /// Common ABI, size, and mutable-state offset header at arena offset zero.
  xpool::arena::LayoutHeader header;
  /// Configuration-order model index used by the Fabric model table.
  std::size_t instance_index;
  /// Rank-local Instance identity within this model topology.
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
  std::size_t max_tokens;
  /// Hidden-state columns in every request.
  std::size_t hidden_size;
  /// Stable xpool::abi::TensorDType value for both payloads.
  std::uint32_t dtype;
  /// Offset of the sole TransportMailbox.
  std::size_t mailbox_offset;
  /// Offset of the fixed input payload byte range.
  std::size_t input_payload_offset;
  /// Offset of the fixed output payload byte range.
  std::size_t output_payload_offset;
  /// Offset of DP token counts, or zero when atn_dp_size is one.
  std::size_t dp_token_counts_offset;
  /// Optional Transport trace buffer geometry.
  xpool::trace::BufferLayout trace;

  /// Plan one complete aligned Transport arena.
  /// \param instance_index Configuration-order model index.
  /// \param instance_rank Rank-local Instance identity.
  /// \param atn_tp_rank Attention tensor-parallel coordinate.
  /// \param atn_tp_size Attention tensor-parallel width.
  /// \param atn_dp_rank Attention data-parallel coordinate.
  /// \param atn_dp_size Attention data-parallel width.
  /// \param max_tokens Maximum physical payload rows.
  /// \param hidden_size Hidden-state columns.
  /// \param dtype Exact hidden-state element type.
  /// \return Validated arena layout with debug-derived trace geometry.
  static TransportArenaLayout create(std::size_t instance_index, std::size_t instance_rank,
                                     std::size_t atn_tp_rank, std::size_t atn_tp_size,
                                     std::size_t atn_dp_rank, std::size_t atn_dp_size,
                                     std::size_t max_tokens, std::size_t hidden_size,
                                     xpool::abi::TensorDType dtype);
  /// Validate every identity, topology, geometry, offset, and total-size fact.
  void validate() const;
  /// Compare complete immutable Transport layouts.
  /// \return True when every layout field is equal.
  constexpr bool operator==(const TransportArenaLayout &) const = default;
};

} // namespace xpool::transport
