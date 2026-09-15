#pragma once

/// \file xpool/kv/channel.hpp
/// \brief Host-local elastic KV-capacity command and report channel.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <xpool/kv/layout.hpp>
#include <xpool/utils/shmem.hpp>

namespace xpool::kv {

/// One coherent group capacity command.
struct KvCapacityCommand {
  /// Nonzero group-local command identity.
  std::uint32_t sequence;
  /// Physical bundle prefix every partition must protect.
  std::uint32_t target_bundles;
  /// Logical bundle prefix currently exposed to allocators.
  std::uint32_t active_bundles;

  bool operator==(const KvCapacityCommand &) const = default;
};

/// One partition's physical progress and protected command prefix.
struct KvCapacityBackingReport {
  /// Command sequence whose target prefix this partition protects, or zero
  /// during bootstrap.
  std::uint32_t prepared_sequence;
  /// Actual contiguous physical bundle prefix.
  std::uint32_t backed_bundles;

  bool operator==(const KvCapacityBackingReport &) const = default;
};

/// Latest capacity-pressure state published by one logical group.
struct KvCapacityPressureReport {
  /// Nonzero group-local pressure publication identity.
  std::uint32_t sequence;
  /// Active capacity at an admission failure, or no value after resolution.
  std::optional<std::uint32_t> active_bundles;

  bool operator==(const KvCapacityPressureReport &) const = default;
};

/// One attention GPU's post-capture memory observation.
struct KvDeviceMemoryReport {
  /// Total device memory in bytes.
  std::uint64_t total_bytes;
  /// Free device memory in bytes at the observation boundary.
  std::uint64_t free_bytes;

  bool operator==(const KvDeviceMemoryReport &) const = default;
};

/// Move-only owner and typed access surface for one validated channel mapping.
class CapacityChannelMapping {
public:
  /// Construct an empty mapping owner.
  CapacityChannelMapping() = default;

  /// Create shared storage and initialize its channel header.
  static CapacityChannelMapping create(std::string name, CapacityChannelLayout layout);

  /// Attach shared storage and validate its channel header and layout.
  static CapacityChannelMapping attach(std::string_view name);

  ~CapacityChannelMapping() noexcept = default;
  CapacityChannelMapping(const CapacityChannelMapping &) = delete;
  CapacityChannelMapping &operator=(const CapacityChannelMapping &) = delete;
  /// Transfer the mapping and optional unlink authority, leaving the source empty.
  CapacityChannelMapping(CapacityChannelMapping &&) noexcept = default;
  /// Close this mapping before taking the source's mapping and optional unlink authority.
  CapacityChannelMapping &operator=(CapacityChannelMapping &&) noexcept = default;

  /// Return the creator-owned name used for participant attachment.
  /// \pre This mapping was returned by create() and has not been closed.
  std::string_view unlink_name() const noexcept { return memory_.unlink_name(); }

  /// Return the mapping's canonical counts and offsets.
  const CapacityChannelLayout &layout() const noexcept { return layout_; }

  /// Return every attention-device observation slot.
  std::span<PoolEntry> pool_entries() const noexcept {
    return {reinterpret_cast<PoolEntry *>(memory_.bytes().data() + layout_.pools_offset), layout_.pool_count};
  }

  /// Return every logical-group command and pressure slot.
  std::span<GroupEntry> group_entries() const noexcept {
    return {reinterpret_cast<GroupEntry *>(memory_.bytes().data() + layout_.groups_offset), layout_.group_count};
  }

  /// Return every Instance-rank capture and backing slot.
  std::span<PartitionEntry> partition_entries() const noexcept {
    return {reinterpret_cast<PartitionEntry *>(memory_.bytes().data() + layout_.partitions_offset),
            layout_.partition_count};
  }

  /// Release this process's mapping and descriptor and unlink owned storage.
  void close() { memory_.close(); }

private:
  CapacityChannelMapping(xpool::utils::PosixSharedMemoryMapping memory, CapacityChannelLayout layout)
      : memory_(std::move(memory)), layout_(layout) {}

  xpool::utils::PosixSharedMemoryMapping memory_;
  CapacityChannelLayout layout_{};
};

/// Daemon-owned writer and aggregate reader for one Generation channel.
class DaemonCapacityChannel {
public:
  /// Create a zero-initialized POSIX shared-memory channel.
  static DaemonCapacityChannel create(std::size_t pool_count, std::size_t group_count, std::size_t partition_count);

  ~DaemonCapacityChannel() = default;
  /// Transfer the mapping and unlink ownership, leaving the source empty.
  DaemonCapacityChannel(DaemonCapacityChannel &&other) noexcept = default;
  /// Close this mapping before taking the source's mapping and unlink ownership.
  DaemonCapacityChannel &operator=(DaemonCapacityChannel &&other) noexcept = default;
  DaemonCapacityChannel(const DaemonCapacityChannel &) = delete;
  DaemonCapacityChannel &operator=(const DaemonCapacityChannel &) = delete;

  /// Return the opaque POSIX shared-memory name used by participants.
  std::string name() const;

  /// Read pool observations in pool-index order.
  std::vector<std::optional<KvDeviceMemoryReport>> read_device_memory() const;

  /// Publish one complete command for a logical group.
  void publish_command(std::size_t group_index, KvCapacityCommand command);

  /// Read partition reports in partition-index order.
  std::vector<std::optional<KvCapacityBackingReport>> read_backing_reports() const;

  /// Read pressure snapshots in group-index order.
  std::vector<std::optional<KvCapacityPressureReport>> read_pressure_reports() const;

  /// Unmap, close, and unlink the channel. Repeated calls are no-ops.
  void close();

private:
  DaemonCapacityChannel() = default;

  CapacityChannelMapping mapping_;
};

/// AtnAgent-owned publisher for one device observation.
class AtnAgentCapacityChannel {
public:
  /// Attach to a Generation channel and bind this AtnAgent's owned slots.
  static AtnAgentCapacityChannel attach(std::string_view name, std::size_t pool_index,
                                        std::vector<std::size_t> partition_indices);

  ~AtnAgentCapacityChannel() = default;
  /// Transfer the local mapping, leaving the source empty.
  AtnAgentCapacityChannel(AtnAgentCapacityChannel &&other) noexcept = default;
  /// Close this mapping before taking the source's local mapping.
  AtnAgentCapacityChannel &operator=(AtnAgentCapacityChannel &&other) noexcept = default;
  AtnAgentCapacityChannel(const AtnAgentCapacityChannel &) = delete;
  AtnAgentCapacityChannel &operator=(const AtnAgentCapacityChannel &) = delete;

  /// Return whether every bound Instance partition completed Graph capture.
  bool captures_complete() const;

  /// Publish the sole post-capture device-memory observation.
  void publish_device_memory(std::uint64_t total_bytes, std::uint64_t free_bytes);

  /// Release this process's local mapping and descriptor.
  void close();

private:
  AtnAgentCapacityChannel() = default;

  CapacityChannelMapping mapping_;
  std::size_t pool_index_ = 0;
  std::vector<std::size_t> partition_indices_;
};

/// Instance-Rank-owned command reader and partition/group publisher.
class InstanceCapacityChannel {
public:
  /// Attach to one Instance row and bind the rank's owned group and partition.
  static InstanceCapacityChannel attach(std::string_view name, std::size_t group_index, std::size_t partition_index,
                                        std::size_t group_count);

  ~InstanceCapacityChannel() = default;
  /// Transfer the local mapping, leaving the source empty.
  InstanceCapacityChannel(InstanceCapacityChannel &&other) noexcept = default;
  /// Close this mapping before taking the source's local mapping.
  InstanceCapacityChannel &operator=(InstanceCapacityChannel &&other) noexcept = default;
  InstanceCapacityChannel(const InstanceCapacityChannel &) = delete;
  InstanceCapacityChannel &operator=(const InstanceCapacityChannel &) = delete;

  /// Publish the one-way Graph-capture completion barrier.
  void publish_capture_complete();

  /// Publish this partition's actual and command-correlated backing state.
  void publish_backing_report(KvCapacityBackingReport report);

  /// Read coherent commands for this Instance's DP groups in DP-rank order.
  std::vector<std::optional<KvCapacityCommand>> read_commands() const;

  /// Publish a new pressure attempt or its resolution.
  void publish_pressure(std::optional<std::uint32_t> active_bundles);

  /// Release this process's local mapping and descriptor.
  void close();

private:
  InstanceCapacityChannel() = default;

  CapacityChannelMapping mapping_;
  std::size_t group_index_ = 0;
  std::size_t group_row_begin_ = 0;
  std::size_t group_count_ = 0;
  std::size_t partition_index_ = 0;
};

} // namespace xpool::kv
