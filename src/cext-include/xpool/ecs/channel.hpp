#pragma once

/// \file xpool/ecs/channel.hpp
/// \brief Host-local Elastic Context State control channel.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <xpool/ecs/layout.hpp>
#include <xpool/utils/shmem.hpp>

namespace xpool::ecs {

/// One immutable group capacity operation.
struct QuotaCommand {
  /// Nonzero group-local operation identity.
  std::uint32_t sequence;
  /// Absolute physical-byte permission in this group's TP-rank order.
  std::vector<std::uint64_t> quota_bytes;

  bool operator==(const QuotaCommand &) const = default;
};

/// Latest admission demand evaluated under one completed operation.
struct QuotaDemand {
  /// Completed operation whose capacity was evaluated.
  std::uint32_t evaluated_sequence;
  /// Absolute per-rank backing requirements, or no value when demand resolved.
  std::optional<std::vector<std::uint64_t>> required_bytes;
  /// Absolute scheduler-local SLO deadline, or no value when demand resolved.
  std::optional<std::uint64_t> deadline_monotonic_ns;

  bool operator==(const QuotaDemand &) const = default;
};

/// One partition's terminal operation result.
struct QuotaCompletion {
  /// Completed operation.
  std::uint32_t sequence;
  /// Actual retained physical bytes after quota adoption and terminal retirement.
  std::uint64_t backed_bytes;

  bool operator==(const QuotaCompletion &) const = default;
};

/// One attention device's post-capture memory observation.
struct DeviceMemoryReport {
  /// Total device memory in bytes.
  std::uint64_t total_bytes;
  /// Free device memory in bytes at the observation boundary.
  std::uint64_t free_bytes;

  bool operator==(const DeviceMemoryReport &) const = default;
};

/// Move-only owner and typed access surface for one validated channel mapping.
class ControlChannelMapping {
public:
  /// Construct an empty mapping owner.
  ControlChannelMapping() = default;

  /// Create shared storage and initialize its channel header.
  static ControlChannelMapping create(std::string name, std::size_t device_count,
                                      std::span<const std::size_t> group_partition_counts);

  /// Attach shared storage and validate its channel header and layout.
  static ControlChannelMapping attach(std::string_view name);

  ~ControlChannelMapping() noexcept = default;
  ControlChannelMapping(const ControlChannelMapping &) = delete;
  ControlChannelMapping &operator=(const ControlChannelMapping &) = delete;
  /// Transfer the mapping and optional unlink authority, leaving the source empty.
  ControlChannelMapping(ControlChannelMapping &&) noexcept = default;
  /// Close this mapping before taking the source's mapping and optional unlink authority.
  ControlChannelMapping &operator=(ControlChannelMapping &&) noexcept = default;

  /// Return the creator-owned name used for participant attachment.
  /// \pre This mapping was returned by create() and has not been closed.
  std::string_view unlink_name() const noexcept { return memory_.unlink_name(); }

  /// Return the mapping's canonical counts and offsets.
  const ControlChannelLayout &layout() const noexcept { return layout_; }

  /// Return every attention-device observation slot.
  std::span<DeviceMemoryEntry> device_entries() const noexcept {
    return {reinterpret_cast<DeviceMemoryEntry *>(memory_.bytes().data() + layout_.devices_offset),
            layout_.device_count};
  }

  /// Return every logical Capacity Group slot.
  std::span<GroupEntry> group_entries() const noexcept {
    return {reinterpret_cast<GroupEntry *>(memory_.bytes().data() + layout_.groups_offset), layout_.group_count};
  }

  /// Return every Instance-rank byte-vector, startup and operation-completion slot.
  std::span<PartitionEntry> partition_entries() const noexcept {
    return {reinterpret_cast<PartitionEntry *>(memory_.bytes().data() + layout_.partitions_offset),
            layout_.partition_count};
  }

  /// Release this process's mapping and descriptor and unlink owned storage.
  void close() { memory_.close(); }

private:
  ControlChannelMapping(xpool::utils::PosixSharedMemoryMapping memory, ControlChannelLayout layout)
      : memory_(std::move(memory)), layout_(layout) {}

  xpool::utils::PosixSharedMemoryMapping memory_;
  ControlChannelLayout layout_{};
};

/// Daemon-owned writer and aggregate reader for one Generation channel.
class DaemonControlChannel {
public:
  /// Create a zero-initialized POSIX shared-memory channel.
  static DaemonControlChannel create(std::size_t device_count, std::vector<std::size_t> group_partition_counts);

  ~DaemonControlChannel() = default;
  /// Transfer channel ownership, leaving the source empty.
  DaemonControlChannel(DaemonControlChannel &&other) noexcept = default;
  /// Replace this channel with the source, leaving the source empty.
  DaemonControlChannel &operator=(DaemonControlChannel &&other) noexcept = default;
  DaemonControlChannel(const DaemonControlChannel &) = delete;
  DaemonControlChannel &operator=(const DaemonControlChannel &) = delete;

  /// Return the opaque POSIX shared-memory name used by participants.
  std::string name() const;

  /// Read physical memory observations in device-index order.
  std::vector<std::optional<DeviceMemoryReport>> read_device_memory() const;

  /// Read one-time startup backing in partition-index order.
  std::vector<std::optional<std::uint64_t>> read_initial_backing() const;

  /// Publish one Capacity Group's immutable service ceiling.
  void publish_service_ceiling(std::size_t group_index, std::vector<std::uint64_t> quota_bytes);

  /// Publish one immutable operation for a Capacity Group.
  void publish_command(std::size_t group_index, QuotaCommand command);

  /// Read latest demand snapshots in group-index order.
  std::vector<std::optional<QuotaDemand>> read_demands() const;

  /// Read terminal partition completions in partition-index order.
  std::vector<std::optional<QuotaCompletion>> read_completions() const;

  /// Unmap, close, and unlink the channel. Repeated calls are no-ops.
  void close();

private:
  DaemonControlChannel() = default;

  ControlChannelMapping mapping_;
};

/// AtnAgent-owned publisher for one device observation.
class AtnAgentControlChannel {
public:
  /// Attach to a Generation channel and bind this AtnAgent's owned slots.
  static AtnAgentControlChannel attach(std::string_view name, std::size_t device_index,
                                       std::vector<std::size_t> partition_indices);

  ~AtnAgentControlChannel() = default;
  /// Transfer the attachment, leaving the source empty.
  AtnAgentControlChannel(AtnAgentControlChannel &&other) noexcept = default;
  /// Replace this attachment with the source, leaving the source empty.
  AtnAgentControlChannel &operator=(AtnAgentControlChannel &&other) noexcept = default;
  AtnAgentControlChannel(const AtnAgentControlChannel &) = delete;
  AtnAgentControlChannel &operator=(const AtnAgentControlChannel &) = delete;

  /// Return whether every bound Instance partition completed Graph capture.
  bool captures_complete() const;

  /// Publish the sole post-capture device-memory observation.
  void publish_device_memory(std::uint64_t total_bytes, std::uint64_t free_bytes);

  /// Release this process's local mapping and descriptor.
  void close();

private:
  AtnAgentControlChannel() = default;

  ControlChannelMapping mapping_;
  std::size_t device_index_ = 0;
  std::vector<std::size_t> partition_indices_;
};

/// Instance-rank-owned group negotiation and partition publication surface.
class InstanceControlChannel {
public:
  /// Attach to one Instance row and bind the rank's group and partition.
  static InstanceControlChannel attach(std::string_view name, std::size_t group_index, std::size_t partition_index,
                                       std::size_t dp_group_count);

  ~InstanceControlChannel() = default;
  /// Transfer the attachment, leaving the source empty.
  InstanceControlChannel(InstanceControlChannel &&other) noexcept = default;
  /// Replace this attachment with the source, leaving the source empty.
  InstanceControlChannel &operator=(InstanceControlChannel &&other) noexcept = default;
  InstanceControlChannel(const InstanceControlChannel &) = delete;
  InstanceControlChannel &operator=(const InstanceControlChannel &) = delete;

  /// Publish immutable startup bytes after Graph capture and device synchronization.
  /// Both daemon startup and AtnAgent capture readiness acquire this publication.
  void publish_initial_backing(std::uint64_t backed_bytes);

  /// Read this Capacity Group's immutable service ceiling.
  std::optional<std::vector<std::uint64_t>> service_ceiling() const;

  /// Read coherent commands for this Instance's DP groups in DP-rank order.
  std::vector<std::optional<QuotaCommand>> read_commands() const;

  /// Publish this partition's terminal operation result.
  void publish_completion(QuotaCompletion completion);

  /// Publish the latest leader-owned admission-demand snapshot.
  void publish_demand(QuotaDemand demand);

  /// Release this process's local mapping and descriptor.
  void close();

private:
  InstanceControlChannel() = default;

  ControlChannelMapping mapping_;
  std::size_t group_index_ = 0;
  std::size_t group_row_begin_ = 0;
  std::size_t dp_group_count_ = 0;
  std::size_t partition_index_ = 0;
};

} // namespace xpool::ecs
