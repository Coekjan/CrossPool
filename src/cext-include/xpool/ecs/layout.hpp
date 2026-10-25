#pragma once

/// \file xpool/ecs/layout.hpp
/// \brief Shared-memory layout for one Generation's Elastic Context State control channel.

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace xpool::ecs {

/// Stable magic identifying an Elastic Context State control channel.
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
  std::uint64_t device_count;
  /// Number of logical capacity groups.
  std::uint64_t group_count;
  /// Number of Instance-rank partition entries.
  std::uint64_t partition_count;

  constexpr bool operator==(const ControlChannelHeader &) const = default;
};

/// One AtnAgent-owned device-memory publication.
struct alignas(std::uint64_t) DeviceMemoryEntry {
  /// Device-total bytes; zero until the AtnAgent publishes the entry.
  std::uint64_t device_total_bytes;
  /// Device-free bytes sampled with the published total.
  std::uint64_t device_free_bytes;
};

/// One Capacity Group's immutable directory and atomic multiword publications.
struct alignas(std::uint64_t) GroupEntry {
  /// First partition in this group's contiguous TP-fastest range.
  std::uint64_t first_partition;
  /// Number of partitions in this group; fixed before mapping publication.
  std::uint64_t partition_count;
  /// Even nonzero command revision; odd while its sole writer publishes.
  std::uint64_t command_revision;
  /// Daemon-owned immutable operation identity.
  std::uint32_t command_sequence;
  /// Release-published readiness of this group's immutable ceiling vector.
  std::uint32_t ceiling_ready;
  /// Even nonzero demand revision; odd while its sole leader publishes.
  std::uint64_t demand_revision;
  /// Completed grant against which the latest work was evaluated.
  std::uint32_t demand_evaluated_sequence;
  /// One when both the requirement vector and deadline are present.
  std::uint32_t demand_present;
  /// Absolute scheduler-local SLO deadline; not a publication sentinel.
  std::uint64_t demand_deadline_monotonic_ns;
};

/// One Instance rank's byte-vector cells, startup and terminal completion.
struct alignas(std::uint64_t) PartitionEntry {
  /// Daemon-owned absolute permission, covered by the group's command revision.
  std::uint64_t command_quota_bytes;
  /// Immutable daemon-owned bound, covered by the group's ceiling readiness.
  std::uint64_t service_ceiling_bytes;
  /// Leader-owned requirement, covered by the group's demand revision.
  std::uint64_t required_bytes;
  /// One-time actual physical startup bytes; zero is a valid value.
  std::uint64_t initial_backing_bytes;
  /// Even nonzero completion revision; odd while the rank publishes.
  std::uint64_t completion_revision;
  /// Actual physical backing after adopting the command and retiring storage.
  std::uint64_t completion_backed_bytes;
  /// Release-published readiness after Graph capture and device synchronization.
  std::uint32_t initial_backing_ready;
  /// Completed immutable group command identity.
  std::uint32_t completion_sequence;
};

/// Canonical counts and region offsets for a control-channel mapping.
struct ControlChannelLayout {
  /// Number of attention-device publication entries.
  std::size_t device_count;
  /// Number of logical capacity-group entries.
  std::size_t group_count;
  /// Number of Instance-rank partition entries.
  std::size_t partition_count;
  /// Byte offset of the device-memory-entry array.
  std::size_t devices_offset;
  /// Byte offset of the group-entry array.
  std::size_t groups_offset;
  /// Byte offset of the partition-entry array.
  std::size_t partitions_offset;
  /// Total bytes occupied by the channel mapping.
  std::size_t total_bytes;

  /// Materialize one complete layout from stable Generation membership.
  static ControlChannelLayout create(std::size_t device_count, std::size_t group_count, std::size_t partition_count);

  /// Validate a mapped header and byte extent against this canonical layout.
  void validate(const ControlChannelHeader &header, std::size_t mapping_bytes) const;

  constexpr bool operator==(const ControlChannelLayout &) const = default;
};

static_assert(std::is_trivially_copyable_v<ControlChannelHeader>);

} // namespace xpool::ecs
