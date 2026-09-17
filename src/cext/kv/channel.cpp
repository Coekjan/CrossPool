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
static_assert(std::atomic_ref<SequencedValue>::is_always_lock_free &&
              alignof(SequencedValue) >= std::atomic_ref<SequencedValue>::required_alignment);

ControlChannelMapping ControlChannelMapping::create(std::string name, ControlChannelLayout layout) {
  auto memory = xpool::utils::PosixSharedMemoryMapping::create(std::move(name), layout.total_bytes);
  *reinterpret_cast<ControlChannelHeader *>(memory.bytes().data()) = ControlChannelHeader{
      .magic = kControlChannelMagic,
      .abi_version = xpool::abi::kVersion,
      .reserved = 0,
      .total_bytes = layout.total_bytes,
      .pool_count = layout.pool_count,
      .group_count = layout.group_count,
      .partition_count = layout.partition_count,
  };
  return {std::move(memory), layout};
}

ControlChannelMapping ControlChannelMapping::attach(std::string_view name) {
  auto memory = xpool::utils::PosixSharedMemoryMapping::attach(name);
  const auto bytes = memory.bytes();
  TORCH_CHECK(bytes.size() >= sizeof(ControlChannelHeader), "xpool kv channel shared memory is truncated");
  const auto &header = *reinterpret_cast<const ControlChannelHeader *>(bytes.data());
  const auto layout = ControlChannelLayout::create(static_cast<std::size_t>(header.pool_count),
                                                   static_cast<std::size_t>(header.group_count),
                                                   static_cast<std::size_t>(header.partition_count));
  layout.validate(header, bytes.size());
  return {std::move(memory), layout};
}

DaemonControlChannel DaemonControlChannel::create(std::size_t pool_count, std::size_t group_count,
                                                  std::size_t partition_count) {
  const auto layout = ControlChannelLayout::create(pool_count, group_count, partition_count);
  static auto next_name = std::atomic<std::uint64_t>{1};
  const auto suffix = next_name.fetch_add(1, std::memory_order_relaxed);
  TORCH_CHECK(suffix != 0, "xpool kv channel name space is exhausted");
  auto channel = DaemonControlChannel{};
  channel.mapping_ =
      ControlChannelMapping::create("/xpool-kv-" + std::to_string(getpid()) + "-" + std::to_string(suffix), layout);
  return channel;
}

std::string DaemonControlChannel::name() const { return std::string{mapping_.unlink_name()}; }

std::vector<std::optional<KvDeviceMemoryReport>> DaemonControlChannel::read_device_memory() const {
  const auto entries = mapping_.pool_entries();
  auto reports = std::vector<std::optional<KvDeviceMemoryReport>>(entries.size());
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

std::vector<std::optional<std::uint32_t>> DaemonControlChannel::read_initial_backing() const {
  const auto entries = mapping_.partition_entries();
  auto backing = std::vector<std::optional<std::uint32_t>>(entries.size());
  for (auto index = std::size_t{0}; index < backing.size(); ++index) {
    const auto bundles = std::atomic_ref{entries[index].initial_backing_bundles}.load(std::memory_order_acquire);
    if (bundles != 0) {
      backing[index] = bundles;
    }
  }
  return backing;
}

void DaemonControlChannel::publish_service_ceiling(std::size_t group_index, std::uint32_t bundles) {
  TORCH_CHECK(group_index < mapping_.layout().group_count && bundles != 0,
              "xpool kv service ceiling values are invalid");
  auto &storage = mapping_.group_entries()[group_index].service_ceiling_bundles;
  const auto previous = std::atomic_ref{storage}.load(std::memory_order_acquire);
  TORCH_CHECK(previous == 0 || previous == bundles, "xpool kv service ceiling is immutable");
  std::atomic_ref{storage}.store(bundles, std::memory_order_release);
}

void DaemonControlChannel::publish_command(std::size_t group_index, KvCapacityCommand command) {
  TORCH_CHECK(group_index < mapping_.layout().group_count && command.sequence != 0 && command.target_bundles != 0,
              "xpool kv command values are invalid");
  auto &storage = mapping_.group_entries()[group_index].command;
  const auto previous = std::atomic_ref{storage}.load(std::memory_order_acquire);
  if (previous.sequence == command.sequence) {
    TORCH_CHECK(previous.value == command.target_bundles, "xpool kv command mutates an existing sequence");
    return;
  }
  TORCH_CHECK(previous.sequence != std::numeric_limits<std::uint32_t>::max() &&
                  command.sequence == previous.sequence + 1,
              "xpool kv command sequence is not the next publication");
  std::atomic_ref{storage}.store(SequencedValue{.sequence = command.sequence, .value = command.target_bundles},
                                 std::memory_order_release);
}

std::vector<std::optional<KvCapacityDemand>> DaemonControlChannel::read_demands() const {
  const auto entries = mapping_.group_entries();
  auto demands = std::vector<std::optional<KvCapacityDemand>>(entries.size());
  for (auto index = std::size_t{0}; index < demands.size(); ++index) {
    auto &entry = entries[index];
    const auto first_revision = std::atomic_ref{entry.demand_revision}.load(std::memory_order_seq_cst);
    if (first_revision == 0 || (first_revision & 1) != 0) {
      continue;
    }
    const auto payload = std::atomic_ref{entry.demand_payload}.load(std::memory_order_seq_cst);
    const auto deadline = std::atomic_ref{entry.demand_deadline}.load(std::memory_order_seq_cst);
    const auto second_revision = std::atomic_ref{entry.demand_revision}.load(std::memory_order_seq_cst);
    if (first_revision == second_revision && payload.sequence != 0) {
      demands[index] = KvCapacityDemand{
          .evaluated_sequence = payload.sequence,
          .requested_bundles = deadline == 1 ? std::nullopt : std::optional{payload.value},
          .deadline_monotonic_ns = deadline == 1 ? std::nullopt : std::optional{deadline - 1},
      };
    }
  }
  return demands;
}

std::vector<std::optional<KvCapacityCompletion>> DaemonControlChannel::read_completions() const {
  const auto entries = mapping_.partition_entries();
  auto completions = std::vector<std::optional<KvCapacityCompletion>>(entries.size());
  for (auto index = std::size_t{0}; index < completions.size(); ++index) {
    const auto completion = std::atomic_ref{entries[index].completion}.load(std::memory_order_acquire);
    if (completion.sequence != 0) {
      completions[index] = KvCapacityCompletion{.sequence = completion.sequence, .backed_bundles = completion.value};
    }
  }
  return completions;
}

void DaemonControlChannel::close() { mapping_.close(); }

AtnAgentControlChannel AtnAgentControlChannel::attach(std::string_view name, std::size_t pool_index,
                                                      std::vector<std::size_t> partition_indices) {
  auto mapping = ControlChannelMapping::attach(name);
  TORCH_CHECK(pool_index < mapping.layout().pool_count, "xpool kv atnagent pool index is out of range");
  TORCH_CHECK(!partition_indices.empty(), "xpool kv atnagent partition indices must not be empty");
  TORCH_CHECK(std::ranges::all_of(partition_indices,
                                  [&](std::size_t index) { return index < mapping.layout().partition_count; }),
              "xpool kv atnagent partition index is out of range");
  auto sorted = partition_indices;
  std::ranges::sort(sorted);
  TORCH_CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
              "xpool kv atnagent partition indices must be unique");
  auto channel = AtnAgentControlChannel{};
  channel.mapping_ = std::move(mapping);
  channel.pool_index_ = pool_index;
  channel.partition_indices_ = std::move(partition_indices);
  return channel;
}

bool AtnAgentControlChannel::captures_complete() const {
  const auto entries = mapping_.partition_entries();
  return std::ranges::all_of(partition_indices_, [&](std::size_t index) {
    return std::atomic_ref{entries[index].capture_complete}.load(std::memory_order_acquire) != 0;
  });
}

void AtnAgentControlChannel::publish_device_memory(std::uint64_t total_bytes, std::uint64_t free_bytes) {
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

void AtnAgentControlChannel::close() { mapping_.close(); }

InstanceControlChannel InstanceControlChannel::attach(std::string_view name, std::size_t group_index,
                                                      std::size_t partition_index, std::size_t dp_group_count) {
  auto mapping = ControlChannelMapping::attach(name);
  const auto &layout = mapping.layout();
  TORCH_CHECK(dp_group_count != 0 && dp_group_count <= layout.pool_count && layout.pool_count % dp_group_count == 0,
              "xpool kv instance DP group count is invalid");
  TORCH_CHECK(group_index < layout.group_count && partition_index < layout.partition_count,
              "xpool kv instance slot index is out of range");
  const auto group_row_begin = group_index - group_index % layout.pool_count;
  const auto dp_rank = group_index - group_row_begin;
  TORCH_CHECK(dp_rank < dp_group_count && group_row_begin + dp_group_count <= layout.group_count,
              "xpool kv instance group slot is outside its DP row");
  TORCH_CHECK(partition_index / layout.pool_count == group_index / layout.pool_count,
              "xpool kv instance group and partition slots belong to different instances");
  const auto tp_size = layout.pool_count / dp_group_count;
  const auto partition_group_begin = group_row_begin + dp_rank * tp_size;
  TORCH_CHECK(partition_index >= partition_group_begin && partition_index < partition_group_begin + tp_size,
              "xpool kv instance partition slot is outside its Capacity Group");
  auto channel = InstanceControlChannel{};
  channel.mapping_ = std::move(mapping);
  channel.group_index_ = group_index;
  channel.group_row_begin_ = group_row_begin;
  channel.dp_group_count_ = dp_group_count;
  channel.partition_index_ = partition_index;
  return channel;
}

void InstanceControlChannel::publish_initial_backing(std::uint32_t bundles) {
  TORCH_CHECK(bundles != 0, "xpool kv initial backing must be positive");
  auto &storage = mapping_.partition_entries()[partition_index_].initial_backing_bundles;
  const auto previous = std::atomic_ref{storage}.load(std::memory_order_acquire);
  TORCH_CHECK(previous == 0 || previous == bundles, "xpool kv initial backing is immutable");
  std::atomic_ref{storage}.store(bundles, std::memory_order_release);
}

void InstanceControlChannel::publish_capture_complete() {
  std::atomic_ref{mapping_.partition_entries()[partition_index_].capture_complete}.store(1, std::memory_order_release);
}

std::optional<std::uint32_t> InstanceControlChannel::service_ceiling() const {
  const auto bundles =
      std::atomic_ref{mapping_.group_entries()[group_index_].service_ceiling_bundles}.load(std::memory_order_acquire);
  return bundles == 0 ? std::nullopt : std::optional{bundles};
}

std::vector<std::optional<KvCapacityCommand>> InstanceControlChannel::read_commands() const {
  const auto entries = mapping_.group_entries();
  auto commands = std::vector<std::optional<KvCapacityCommand>>(dp_group_count_);
  for (auto offset = std::size_t{0}; offset < commands.size(); ++offset) {
    const auto publication =
        std::atomic_ref{entries[group_row_begin_ + offset].command}.load(std::memory_order_acquire);
    if (publication.sequence != 0) {
      commands[offset] = KvCapacityCommand{.sequence = publication.sequence, .target_bundles = publication.value};
    }
  }
  return commands;
}

void InstanceControlChannel::publish_completion(KvCapacityCompletion completion) {
  TORCH_CHECK(completion.sequence != 0 && completion.backed_bundles != 0, "xpool kv completion values are invalid");
  auto &storage = mapping_.partition_entries()[partition_index_].completion;
  const auto previous = std::atomic_ref{storage}.load(std::memory_order_acquire);
  TORCH_CHECK(previous.sequence <= completion.sequence, "xpool kv completion sequence regressed");
  TORCH_CHECK(previous.sequence != completion.sequence || previous.value == completion.backed_bundles,
              "xpool kv completion mutates an existing sequence");
  std::atomic_ref{storage}.store(SequencedValue{.sequence = completion.sequence, .value = completion.backed_bundles},
                                 std::memory_order_release);
}

void InstanceControlChannel::publish_demand(KvCapacityDemand demand) {
  TORCH_CHECK(demand.evaluated_sequence != 0 &&
                  demand.requested_bundles.has_value() == demand.deadline_monotonic_ns.has_value() &&
                  (!demand.requested_bundles || *demand.requested_bundles != 0) &&
                  (!demand.deadline_monotonic_ns ||
                   (*demand.deadline_monotonic_ns != 0 &&
                    *demand.deadline_monotonic_ns != std::numeric_limits<std::uint64_t>::max())),
              "xpool kv demand values are invalid");
  auto &entry = mapping_.group_entries()[group_index_];
  const auto previous = std::atomic_ref{entry.demand_payload}.load(std::memory_order_acquire);
  TORCH_CHECK(previous.sequence <= demand.evaluated_sequence, "xpool kv demand evaluated sequence regressed");
  auto revision = std::atomic_ref{entry.demand_revision};
  revision.fetch_add(1, std::memory_order_seq_cst);
  std::atomic_ref{entry.demand_payload}.store(
      SequencedValue{.sequence = demand.evaluated_sequence, .value = demand.requested_bundles.value_or(0)},
      std::memory_order_seq_cst);
  std::atomic_ref{entry.demand_deadline}.store(demand.deadline_monotonic_ns ? *demand.deadline_monotonic_ns + 1 : 1,
                                               std::memory_order_seq_cst);
  revision.fetch_add(1, std::memory_order_seq_cst);
}

void InstanceControlChannel::close() { mapping_.close(); }

} // namespace xpool::kv
