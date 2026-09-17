#pragma once

/// \file xpool/kv/layout.hpp
/// \brief Shared-memory layout for one Generation's elastic KV control channel.

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace xpool::kv {

/// Stable magic identifying an elastic KV control channel.
inline constexpr std::uint64_t kControlChannelMagic = 0x58504f4f4c4b5631ULL;

/// Fixed prefix stored at offset zero of a control-channel mapping.
struct alignas(std::uint64_t) ControlChannelHeader {
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
  /// Number of Instance-rank partition entries.
  std::uint64_t partition_count;

  constexpr bool operator==(const ControlChannelHeader &) const = default;
};

/// One AtnAgent-owned device-memory publication.
struct alignas(std::uint64_t) PoolEntry {
  /// Device-total bytes; zero until the AtnAgent publishes the entry.
  std::uint64_t device_total_bytes;
  /// Device-free bytes sampled with the published total.
  std::uint64_t device_free_bytes;
};

/// One sequence and value published as a coherent atomic value.
struct alignas(std::uint64_t) SequencedValue {
  /// Monotonic identity within the owning publication stream.
  std::uint32_t sequence;
  /// Domain-specific value associated with the sequence.
  std::uint32_t value;

  constexpr bool operator==(const SequencedValue &) const = default;
};

static_assert(sizeof(SequencedValue) == sizeof(std::uint64_t));

/// One Capacity Group's command, demand, and service ceiling.
struct alignas(std::uint64_t) GroupEntry {
  /// Daemon-owned fixed operation sequence and target bundle count.
  SequencedValue command;
  /// TP-leader-owned evaluated sequence and absolute requested bundle count.
  SequencedValue demand_payload;
  /// Encoded deadline; one means resolved demand.
  std::uint64_t demand_deadline;
  /// Even nonzero values commit a coherent demand; odd values mark a writer in progress.
  std::uint64_t demand_revision;
  /// Daemon-owned immutable service ceiling; zero until published.
  std::uint32_t service_ceiling_bundles;
};

/// One Instance-rank's startup publications and terminal operation completion.
struct alignas(std::uint64_t) PartitionEntry {
  /// One-time startup physical backing count; zero until published.
  std::uint32_t initial_backing_bundles;
  /// One-way post-capture completion publication.
  std::uint32_t capture_complete;
  /// Completed operation sequence and actual physical backing count.
  SequencedValue completion;
};

/// Canonical counts and region offsets for a control-channel mapping.
struct ControlChannelLayout {
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
  static ControlChannelLayout create(std::size_t pool_count, std::size_t group_count, std::size_t partition_count);

  /// Validate a mapped header and byte extent against this canonical layout.
  void validate(const ControlChannelHeader &header, std::size_t mapping_bytes) const;

  constexpr bool operator==(const ControlChannelLayout &) const = default;
};

static_assert(std::is_trivially_copyable_v<ControlChannelHeader>);

} // namespace xpool::kv
