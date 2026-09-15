#pragma once

/// \file xpool/kv/layout.hpp
/// \brief Shared-memory layout for one Generation's elastic KV-capacity channel.

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace xpool::kv {

/// Stable magic identifying an elastic KV-capacity channel.
inline constexpr std::uint64_t kCapacityChannelMagic = 0x58504f4f4c4b5631ULL;

/// Fixed prefix stored at offset zero of a capacity-channel mapping.
struct alignas(std::uint64_t) CapacityChannelHeader {
  /// Domain-specific channel identity.
  std::uint64_t magic;
  /// Native ABI version required to interpret the mapping.
  std::uint32_t abi_version;
  /// Reserved for natural alignment; must be zero.
  std::uint32_t reserved;
  /// Total bytes occupied by the mapping.
  std::uint64_t total_bytes;
  /// Number of attention-device observations.
  std::uint64_t pool_count;
  /// Number of logical capacity groups.
  std::uint64_t group_count;
  /// Number of Instance-rank backing reports.
  std::uint64_t partition_count;

  constexpr bool operator==(const CapacityChannelHeader &) const = default;
};

/// One AtnAgent-owned device-memory publication.
struct alignas(std::uint64_t) PoolEntry {
  /// Device-total bytes; zero until the AtnAgent publishes the entry.
  std::uint64_t device_total_bytes;
  /// Device-free bytes sampled with the published total.
  std::uint64_t device_free_bytes;
};

/// One sequence and bundle count published as a coherent atomic value.
struct alignas(std::uint64_t) CapacityPublication {
  /// Monotonic identity within the owning publication stream.
  std::uint32_t sequence;
  /// Bundle count associated with the sequence.
  std::uint32_t bundles;

  constexpr bool operator==(const CapacityPublication &) const = default;
};

static_assert(sizeof(CapacityPublication) == sizeof(std::uint64_t));

/// One daemon-owned command and Instance-owned pressure publication.
struct alignas(std::uint64_t) GroupEntry {
  /// Command sequence and target bundle count.
  CapacityPublication target_publication;
  /// Command sequence and active bundle count.
  CapacityPublication active_publication;
  /// Pressure sequence and observed active bundle count.
  CapacityPublication pressure_publication;
};

/// One Instance-rank-owned capture and backing publication.
struct alignas(std::uint64_t) PartitionEntry {
  /// Prepared command sequence and actual backing count.
  CapacityPublication backing_publication;
  /// One-way post-capture completion publication.
  std::uint32_t capture_complete;
};

/// Canonical counts and region offsets for a capacity-channel mapping.
struct CapacityChannelLayout {
  /// Number of attention-device publication entries.
  std::size_t pool_count;
  /// Number of logical capacity-group entries.
  std::size_t group_count;
  /// Number of Instance-rank partition entries.
  std::size_t partition_count;
  /// Byte offset of the pool-entry array.
  std::size_t pools_offset;
  /// Byte offset of the group-entry array.
  std::size_t groups_offset;
  /// Byte offset of the partition-entry array.
  std::size_t partitions_offset;
  /// Total bytes occupied by the channel mapping.
  std::size_t total_bytes;

  /// Materialize one complete layout from stable Generation membership.
  static CapacityChannelLayout create(std::size_t pool_count, std::size_t group_count, std::size_t partition_count);

  /// Validate a mapped header and byte extent against this canonical layout.
  void validate(const CapacityChannelHeader &header, std::size_t mapping_bytes) const;

  constexpr bool operator==(const CapacityChannelLayout &) const = default;
};

static_assert(std::is_trivially_copyable_v<CapacityChannelHeader>);

} // namespace xpool::kv
