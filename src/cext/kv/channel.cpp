#include <xpool/kv/channel.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <c10/util/Exception.h>
#include <unistd.h>

#include <xpool/abi.hpp>

namespace xpool::kv {

static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free &&
              alignof(std::uint64_t) >= std::atomic_ref<std::uint64_t>::required_alignment);
static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free &&
              alignof(std::uint32_t) >= std::atomic_ref<std::uint32_t>::required_alignment);
static_assert(std::atomic_ref<CapacityPublication>::is_always_lock_free &&
              alignof(CapacityPublication) >= std::atomic_ref<CapacityPublication>::required_alignment);

CapacityChannelMapping CapacityChannelMapping::create(std::string name, CapacityChannelLayout layout) {
  auto memory = xpool::utils::PosixSharedMemoryMapping::create(std::move(name), layout.total_bytes);
  *reinterpret_cast<CapacityChannelHeader *>(memory.bytes().data()) = CapacityChannelHeader{
      .magic = kCapacityChannelMagic,
      .abi_version = xpool::abi::kVersion,
      .reserved = 0,
      .total_bytes = layout.total_bytes,
      .pool_count = layout.pool_count,
      .group_count = layout.group_count,
      .partition_count = layout.partition_count,
  };
  return {std::move(memory), layout};
}

CapacityChannelMapping CapacityChannelMapping::attach(std::string_view name) {
  auto memory = xpool::utils::PosixSharedMemoryMapping::attach(name);
  const auto bytes = memory.bytes();
  TORCH_CHECK(bytes.size() >= sizeof(CapacityChannelHeader), "xpool kv channel shared memory is truncated");
  const auto &header = *reinterpret_cast<const CapacityChannelHeader *>(bytes.data());
  const auto layout = CapacityChannelLayout::create(static_cast<std::size_t>(header.pool_count),
                                                    static_cast<std::size_t>(header.group_count),
                                                    static_cast<std::size_t>(header.partition_count));
  layout.validate(header, bytes.size());
  return {std::move(memory), layout};
}

DaemonCapacityChannel DaemonCapacityChannel::create(std::size_t pool_count, std::size_t group_count,
                                                    std::size_t partition_count) {
  const auto layout = CapacityChannelLayout::create(pool_count, group_count, partition_count);
  static auto next_name = std::atomic<std::uint64_t>{1};
  const auto suffix = next_name.fetch_add(1, std::memory_order_relaxed);
  TORCH_CHECK(suffix != 0, "xpool kv channel name space is exhausted");
  auto channel = DaemonCapacityChannel{};
  channel.mapping_ =
      CapacityChannelMapping::create("/xpool-kv-" + std::to_string(getpid()) + "-" + std::to_string(suffix), layout);
  return channel;
}

std::string DaemonCapacityChannel::name() const { return std::string{mapping_.unlink_name()}; }

std::vector<std::optional<KvDeviceMemoryReport>> DaemonCapacityChannel::read_device_memory() const {
  const auto &channel = mapping_;
  const auto entries = channel.pool_entries();
  auto reports = std::vector<std::optional<KvDeviceMemoryReport>>(channel.layout().pool_count);
  for (auto index = std::size_t{0}; index < reports.size(); ++index) {
    const auto total = std::atomic_ref{entries[index].device_total_bytes}.load(std::memory_order_acquire);
    if (total != 0) {
      reports[index] = KvDeviceMemoryReport{
          .total_bytes = total,
          .free_bytes = std::atomic_ref{entries[index].device_free_bytes}.load(std::memory_order_relaxed),
      };
    }
  }
  return reports;
}

void DaemonCapacityChannel::publish_command(std::size_t group_index, KvCapacityCommand command) {
  const auto &channel = mapping_;
  TORCH_CHECK(group_index < channel.layout().group_count, "xpool kv command group index is out of range");
  TORCH_CHECK(command.sequence != 0 && command.target_bundles != 0 && command.active_bundles != 0 &&
                  command.active_bundles <= command.target_bundles,
              "xpool kv command values are invalid");
  auto &entry = channel.group_entries()[group_index];
  const auto previous_target = std::atomic_ref{entry.target_publication}.load(std::memory_order_acquire);
  const auto previous_active = std::atomic_ref{entry.active_publication}.load(std::memory_order_acquire);
  const auto previous_sequence = previous_target.sequence;
  if (previous_sequence == command.sequence) {
    TORCH_CHECK(previous_target.bundles == command.target_bundles && previous_active.sequence == command.sequence &&
                    command.active_bundles >= previous_active.bundles,
                "xpool kv command mutates an existing sequence");
  } else {
    TORCH_CHECK(previous_sequence != std::numeric_limits<std::uint32_t>::max() &&
                    command.sequence == previous_sequence + 1,
                "xpool kv command sequence is not the next publication");
  }
  std::atomic_ref{entry.target_publication}.store(
      CapacityPublication{.sequence = command.sequence, .bundles = command.target_bundles}, std::memory_order_release);
  std::atomic_ref{entry.active_publication}.store(
      CapacityPublication{.sequence = command.sequence, .bundles = command.active_bundles}, std::memory_order_release);
}

std::vector<std::optional<KvCapacityBackingReport>> DaemonCapacityChannel::read_backing_reports() const {
  const auto &channel = mapping_;
  const auto entries = channel.partition_entries();
  auto reports = std::vector<std::optional<KvCapacityBackingReport>>(channel.layout().partition_count);
  for (auto index = std::size_t{0}; index < reports.size(); ++index) {
    const auto publication = std::atomic_ref{entries[index].backing_publication}.load(std::memory_order_acquire);
    if (publication.bundles != 0) {
      reports[index] = KvCapacityBackingReport{
          .prepared_sequence = publication.sequence,
          .backed_bundles = publication.bundles,
      };
    }
  }
  return reports;
}

std::vector<std::optional<KvCapacityPressureReport>> DaemonCapacityChannel::read_pressure_reports() const {
  const auto &channel = mapping_;
  const auto entries = channel.group_entries();
  auto reports = std::vector<std::optional<KvCapacityPressureReport>>(channel.layout().group_count);
  for (auto index = std::size_t{0}; index < reports.size(); ++index) {
    const auto publication = std::atomic_ref{entries[index].pressure_publication}.load(std::memory_order_acquire);
    const auto sequence = publication.sequence;
    if (sequence != 0) {
      const auto active_bundles = publication.bundles;
      reports[index] = KvCapacityPressureReport{
          .sequence = sequence,
          .active_bundles = active_bundles == 0 ? std::nullopt : std::optional{active_bundles},
      };
    }
  }
  return reports;
}

void DaemonCapacityChannel::close() { mapping_.close(); }

AtnAgentCapacityChannel AtnAgentCapacityChannel::attach(std::string_view name, std::size_t pool_index,
                                                        std::vector<std::size_t> partition_indices) {
  auto mapping = CapacityChannelMapping::attach(name);
  TORCH_CHECK(pool_index < mapping.layout().pool_count, "xpool kv atnagent pool index is out of range");
  TORCH_CHECK(!partition_indices.empty(), "xpool kv atnagent partition indices must not be empty");
  TORCH_CHECK(std::ranges::all_of(partition_indices,
                                  [&](std::size_t index) { return index < mapping.layout().partition_count; }),
              "xpool kv atnagent partition index is out of range");
  auto sorted = partition_indices;
  std::ranges::sort(sorted);
  TORCH_CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
              "xpool kv atnagent partition indices must be unique");
  auto channel = AtnAgentCapacityChannel{};
  channel.mapping_ = std::move(mapping);
  channel.pool_index_ = pool_index;
  channel.partition_indices_ = std::move(partition_indices);
  return channel;
}

bool AtnAgentCapacityChannel::captures_complete() const {
  const auto entries = mapping_.partition_entries();
  return std::ranges::all_of(partition_indices_, [&](std::size_t index) {
    return std::atomic_ref{entries[index].capture_complete}.load(std::memory_order_acquire) != 0;
  });
}

void AtnAgentCapacityChannel::publish_device_memory(std::uint64_t total_bytes, std::uint64_t free_bytes) {
  TORCH_CHECK(total_bytes != 0 && free_bytes <= total_bytes, "xpool kv device-memory observation is invalid");
  auto &entry = mapping_.pool_entries()[pool_index_];
  const auto published_total = std::atomic_ref{entry.device_total_bytes}.load(std::memory_order_acquire);
  if (published_total != 0) {
    TORCH_CHECK(published_total == total_bytes &&
                    std::atomic_ref{entry.device_free_bytes}.load(std::memory_order_relaxed) == free_bytes,
                "xpool kv device-memory observation was already published with different values");
    return;
  }
  std::atomic_ref{entry.device_free_bytes}.store(free_bytes, std::memory_order_relaxed);
  std::atomic_ref{entry.device_total_bytes}.store(total_bytes, std::memory_order_release);
}

void AtnAgentCapacityChannel::close() { mapping_.close(); }

InstanceCapacityChannel InstanceCapacityChannel::attach(std::string_view name, std::size_t group_index,
                                                        std::size_t partition_index, std::size_t group_count) {
  auto mapping = CapacityChannelMapping::attach(name);
  const auto &layout = mapping.layout();
  TORCH_CHECK(group_count != 0 && group_count <= layout.pool_count, "xpool kv instance group count is out of range");
  TORCH_CHECK(group_index < layout.group_count && partition_index < layout.partition_count,
              "xpool kv instance slot index is out of range");
  const auto group_row_begin = group_index - group_index % layout.pool_count;
  TORCH_CHECK(group_index - group_row_begin < group_count && group_row_begin + group_count <= layout.group_count,
              "xpool kv instance group slot is outside its effective DP row");
  TORCH_CHECK(partition_index / layout.pool_count == group_index / layout.pool_count,
              "xpool kv instance group and partition slots belong to different instances");
  auto channel = InstanceCapacityChannel{};
  channel.mapping_ = std::move(mapping);
  channel.group_index_ = group_index;
  channel.group_row_begin_ = group_row_begin;
  channel.group_count_ = group_count;
  channel.partition_index_ = partition_index;
  return channel;
}

void InstanceCapacityChannel::publish_capture_complete() {
  std::atomic_ref{mapping_.partition_entries()[partition_index_].capture_complete}.store(1, std::memory_order_release);
}

void InstanceCapacityChannel::publish_backing_report(KvCapacityBackingReport report) {
  TORCH_CHECK(report.backed_bundles != 0, "xpool kv backing report must retain a positive bundle prefix");
  auto &publication = mapping_.partition_entries()[partition_index_].backing_publication;
  const auto previous = std::atomic_ref{publication}.load(std::memory_order_acquire);
  TORCH_CHECK(previous.sequence <= report.prepared_sequence, "xpool kv backing report prepared sequence regressed");
  std::atomic_ref{publication}.store(
      CapacityPublication{.sequence = report.prepared_sequence, .bundles = report.backed_bundles},
      std::memory_order_release);
}

std::vector<std::optional<KvCapacityCommand>> InstanceCapacityChannel::read_commands() const {
  const auto entries = mapping_.group_entries();
  auto commands = std::vector<std::optional<KvCapacityCommand>>(group_count_);
  for (auto offset = std::size_t{0}; offset < group_count_; ++offset) {
    const auto &entry = entries[group_row_begin_ + offset];
    const auto first_target = std::atomic_ref{entry.target_publication}.load(std::memory_order_acquire);
    const auto active = std::atomic_ref{entry.active_publication}.load(std::memory_order_acquire);
    const auto second_target = std::atomic_ref{entry.target_publication}.load(std::memory_order_acquire);
    const auto sequence = first_target.sequence;
    if (first_target == second_target && sequence != 0 && active.sequence == sequence) {
      commands[offset] = KvCapacityCommand{
          .sequence = sequence,
          .target_bundles = first_target.bundles,
          .active_bundles = active.bundles,
      };
    }
  }
  return commands;
}

void InstanceCapacityChannel::publish_pressure(std::optional<std::uint32_t> active_bundles) {
  TORCH_CHECK(!active_bundles.has_value() || *active_bundles != 0,
              "xpool kv pressure capacity must be positive when present");
  auto &publication = mapping_.group_entries()[group_index_].pressure_publication;
  const auto previous = std::atomic_ref{publication}.load(std::memory_order_acquire);
  const auto previous_sequence = previous.sequence;
  TORCH_CHECK(previous_sequence != std::numeric_limits<std::uint32_t>::max(),
              "xpool kv pressure sequence is exhausted");
  std::atomic_ref{publication}.store(
      CapacityPublication{.sequence = previous_sequence + 1, .bundles = active_bundles.value_or(0)},
      std::memory_order_release);
}

void InstanceCapacityChannel::close() { mapping_.close(); }

} // namespace xpool::kv
