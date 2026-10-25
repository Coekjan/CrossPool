#include <xpool/ecs/channel.hpp>

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

namespace xpool::ecs {

static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free &&
              alignof(std::uint64_t) >= std::atomic_ref<std::uint64_t>::required_alignment);
static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free &&
              alignof(std::uint32_t) >= std::atomic_ref<std::uint32_t>::required_alignment);

namespace {

std::uint64_t begin_publication(std::uint64_t &storage) {
  auto revision = std::atomic_ref{storage};
  const auto previous = revision.load(std::memory_order_seq_cst);
  TORCH_CHECK((previous & 1) == 0 && previous <= std::numeric_limits<std::uint64_t>::max() - 2,
              "xpool elastic context state publication revision is invalid or exhausted");
  revision.store(previous + 1, std::memory_order_seq_cst);
  return previous + 2;
}

std::optional<QuotaCommand> read_command(const ControlChannelMapping &mapping, std::size_t group_index) {
  auto &entry = mapping.group_entries()[group_index];
  const auto first = std::atomic_ref{entry.command_revision}.load(std::memory_order_seq_cst);
  if (first == 0 || (first & 1) != 0) {
    return std::nullopt;
  }
  auto command = QuotaCommand{
      .sequence = std::atomic_ref{entry.command_sequence}.load(std::memory_order_seq_cst),
      .quota_bytes = {},
  };
  for (auto &partition : mapping.partition_entries().subspan(entry.first_partition, entry.partition_count)) {
    command.quota_bytes.push_back(std::atomic_ref{partition.command_quota_bytes}.load(std::memory_order_seq_cst));
  }
  return first == std::atomic_ref{entry.command_revision}.load(std::memory_order_seq_cst) && command.sequence != 0
             ? std::optional{std::move(command)}
             : std::nullopt;
}

std::optional<QuotaCompletion> read_completion(PartitionEntry &entry) {
  const auto first = std::atomic_ref{entry.completion_revision}.load(std::memory_order_seq_cst);
  if (first == 0 || (first & 1) != 0) {
    return std::nullopt;
  }
  const auto completion = QuotaCompletion{
      .sequence = std::atomic_ref{entry.completion_sequence}.load(std::memory_order_seq_cst),
      .backed_bytes = std::atomic_ref{entry.completion_backed_bytes}.load(std::memory_order_seq_cst),
  };
  return first == std::atomic_ref{entry.completion_revision}.load(std::memory_order_seq_cst) && completion.sequence != 0
             ? std::optional{completion}
             : std::nullopt;
}

} // namespace

ControlChannelMapping ControlChannelMapping::create(std::string name, std::size_t device_count,
                                                    std::span<const std::size_t> group_partition_counts) {
  auto partition_count = std::size_t{0};
  for (const auto count : group_partition_counts) {
    TORCH_CHECK(count != 0 && count <= std::numeric_limits<std::size_t>::max() - partition_count,
                "xpool elastic context state group partition count is invalid");
    partition_count += count;
  }
  const auto layout = ControlChannelLayout::create(device_count, group_partition_counts.size(), partition_count);
  auto memory = xpool::utils::PosixSharedMemoryMapping::create(std::move(name), layout.total_bytes);
  *reinterpret_cast<ControlChannelHeader *>(memory.bytes().data()) = ControlChannelHeader{
      .magic = kControlChannelMagic,
      .abi_version = xpool::abi::kVersion,
      .reserved = 0,
      .total_bytes = layout.total_bytes,
      .device_count = layout.device_count,
      .group_count = layout.group_count,
      .partition_count = layout.partition_count,
  };
  auto mapping = ControlChannelMapping{std::move(memory), layout};
  auto first_partition = std::size_t{0};
  for (auto index = std::size_t{0}; index < group_partition_counts.size(); ++index) {
    auto &entry = mapping.group_entries()[index];
    entry.first_partition = first_partition;
    entry.partition_count = group_partition_counts[index];
    first_partition += group_partition_counts[index];
  }
  return mapping;
}

ControlChannelMapping ControlChannelMapping::attach(std::string_view name) {
  auto memory = xpool::utils::PosixSharedMemoryMapping::attach(name);
  const auto bytes = memory.bytes();
  TORCH_CHECK(bytes.size() >= sizeof(ControlChannelHeader),
              "xpool elastic context state channel shared memory is truncated");
  const auto &header = *reinterpret_cast<const ControlChannelHeader *>(bytes.data());
  const auto layout = ControlChannelLayout::create(static_cast<std::size_t>(header.device_count),
                                                   static_cast<std::size_t>(header.group_count),
                                                   static_cast<std::size_t>(header.partition_count));
  layout.validate(header, bytes.size());
  auto mapping = ControlChannelMapping{std::move(memory), layout};
  auto first_partition = std::size_t{0};
  for (const auto &entry : mapping.group_entries()) {
    TORCH_CHECK(entry.first_partition == first_partition && entry.partition_count != 0 &&
                    entry.partition_count <= layout.partition_count - first_partition,
                "xpool elastic context state channel group directory is invalid");
    first_partition += entry.partition_count;
  }
  TORCH_CHECK(first_partition == layout.partition_count,
              "xpool elastic context state group directory does not cover partitions");
  return mapping;
}

DaemonControlChannel DaemonControlChannel::create(std::size_t device_count,
                                                  std::vector<std::size_t> group_partition_counts) {
  static auto next_name = std::atomic<std::uint64_t>{1};
  const auto suffix = next_name.fetch_add(1, std::memory_order_relaxed);
  TORCH_CHECK(suffix != 0, "xpool elastic context state channel name space is exhausted");
  auto channel = DaemonControlChannel{};
  channel.mapping_ = ControlChannelMapping::create(
      "/xpool-ecs-" + std::to_string(getpid()) + "-" + std::to_string(suffix), device_count, group_partition_counts);
  return channel;
}

std::string DaemonControlChannel::name() const { return std::string{mapping_.unlink_name()}; }

std::vector<std::optional<DeviceMemoryReport>> DaemonControlChannel::read_device_memory() const {
  const auto entries = mapping_.device_entries();
  auto reports = std::vector<std::optional<DeviceMemoryReport>>(entries.size());
  for (auto index = std::size_t{0}; index < reports.size(); ++index) {
    const auto total = std::atomic_ref{entries[index].device_total_bytes}.load(std::memory_order_acquire);
    if (total != 0) {
      reports[index] = DeviceMemoryReport{
          .total_bytes = total,
          .free_bytes = std::atomic_ref{entries[index].device_free_bytes}.load(std::memory_order_relaxed),
      };
    }
  }
  return reports;
}

std::vector<std::optional<std::uint64_t>> DaemonControlChannel::read_initial_backing() const {
  const auto entries = mapping_.partition_entries();
  auto backing = std::vector<std::optional<std::uint64_t>>(entries.size());
  for (auto index = std::size_t{0}; index < backing.size(); ++index) {
    auto &entry = entries[index];
    if (std::atomic_ref{entry.initial_backing_ready}.load(std::memory_order_acquire) != 0) {
      backing[index] = std::atomic_ref{entry.initial_backing_bytes}.load(std::memory_order_relaxed);
    }
  }
  return backing;
}

void DaemonControlChannel::publish_service_ceiling(std::size_t group_index, std::vector<std::uint64_t> quota_bytes) {
  TORCH_CHECK(group_index < mapping_.layout().group_count,
              "xpool elastic context state service ceiling group is invalid");
  auto &entry = mapping_.group_entries()[group_index];
  auto partitions = mapping_.partition_entries().subspan(entry.first_partition, entry.partition_count);
  TORCH_CHECK(quota_bytes.size() == partitions.size(), "xpool elastic context state ceiling vector extent is invalid");
  const auto published = std::atomic_ref{entry.ceiling_ready}.load(std::memory_order_acquire) != 0;
  for (auto index = std::size_t{0}; index < partitions.size(); ++index) {
    if (published) {
      TORCH_CHECK(std::atomic_ref{partitions[index].service_ceiling_bytes}.load(std::memory_order_relaxed) ==
                      quota_bytes[index],
                  "xpool elastic context state service ceiling is immutable");
    } else {
      std::atomic_ref{partitions[index].service_ceiling_bytes}.store(quota_bytes[index], std::memory_order_relaxed);
    }
  }
  std::atomic_ref{entry.ceiling_ready}.store(1, std::memory_order_release);
}

void DaemonControlChannel::publish_command(std::size_t group_index, QuotaCommand command) {
  TORCH_CHECK(group_index < mapping_.layout().group_count && command.sequence != 0,
              "xpool elastic context state command identity is invalid");
  auto &entry = mapping_.group_entries()[group_index];
  auto partitions = mapping_.partition_entries().subspan(entry.first_partition, entry.partition_count);
  TORCH_CHECK(command.quota_bytes.size() == partitions.size(),
              "xpool elastic context state command vector extent is invalid");
  const auto previous = read_command(mapping_, group_index);
  if (previous && previous->sequence == command.sequence) {
    TORCH_CHECK(*previous == command, "xpool elastic context state command mutates an existing sequence");
    return;
  }
  const auto sequence = previous ? previous->sequence : 0;
  TORCH_CHECK(sequence != std::numeric_limits<std::uint32_t>::max() && command.sequence == sequence + 1,
              "xpool elastic context state command sequence is not the next publication");
  if (sequence != 0) {
    for (auto &partition : partitions) {
      const auto completion = read_completion(partition);
      TORCH_CHECK(completion && completion->sequence == sequence,
                  "xpool elastic context state command supersedes an unfinished group operation");
    }
  }
  const auto revision = begin_publication(entry.command_revision);
  std::atomic_ref{entry.command_sequence}.store(command.sequence, std::memory_order_seq_cst);
  for (auto index = std::size_t{0}; index < partitions.size(); ++index) {
    std::atomic_ref{partitions[index].command_quota_bytes}.store(command.quota_bytes[index], std::memory_order_seq_cst);
  }
  std::atomic_ref{entry.command_revision}.store(revision, std::memory_order_seq_cst);
}

std::vector<std::optional<QuotaDemand>> DaemonControlChannel::read_demands() const {
  const auto entries = mapping_.group_entries();
  auto demands = std::vector<std::optional<QuotaDemand>>(entries.size());
  for (auto index = std::size_t{0}; index < demands.size(); ++index) {
    auto &entry = entries[index];
    const auto first = std::atomic_ref{entry.demand_revision}.load(std::memory_order_seq_cst);
    if (first == 0 || (first & 1) != 0) {
      continue;
    }
    const auto sequence = std::atomic_ref{entry.demand_evaluated_sequence}.load(std::memory_order_seq_cst);
    const auto present = std::atomic_ref{entry.demand_present}.load(std::memory_order_seq_cst);
    const auto deadline = std::atomic_ref{entry.demand_deadline_monotonic_ns}.load(std::memory_order_seq_cst);
    auto required = std::vector<std::uint64_t>{};
    for (auto &partition : mapping_.partition_entries().subspan(entry.first_partition, entry.partition_count)) {
      required.push_back(std::atomic_ref{partition.required_bytes}.load(std::memory_order_seq_cst));
    }
    if (first == std::atomic_ref{entry.demand_revision}.load(std::memory_order_seq_cst) && sequence != 0) {
      TORCH_CHECK(present <= 1, "xpool elastic context state demand presence is invalid");
      demands[index] = QuotaDemand{
          .evaluated_sequence = sequence,
          .required_bytes = present ? std::optional{std::move(required)} : std::nullopt,
          .deadline_monotonic_ns = present ? std::optional{deadline} : std::nullopt,
      };
    }
  }
  return demands;
}

std::vector<std::optional<QuotaCompletion>> DaemonControlChannel::read_completions() const {
  auto completions = std::vector<std::optional<QuotaCompletion>>{};
  for (auto &entry : mapping_.partition_entries()) {
    completions.push_back(read_completion(entry));
  }
  return completions;
}

void DaemonControlChannel::close() { mapping_.close(); }

AtnAgentControlChannel AtnAgentControlChannel::attach(std::string_view name, std::size_t device_index,
                                                      std::vector<std::size_t> partition_indices) {
  auto mapping = ControlChannelMapping::attach(name);
  TORCH_CHECK(device_index < mapping.layout().device_count,
              "xpool elastic context state atnagent device index is out of range");
  TORCH_CHECK(!partition_indices.empty(), "xpool elastic context state atnagent partition indices must not be empty");
  TORCH_CHECK(std::ranges::all_of(partition_indices,
                                  [&](std::size_t index) { return index < mapping.layout().partition_count; }),
              "xpool elastic context state atnagent partition index is out of range");
  auto sorted = partition_indices;
  std::ranges::sort(sorted);
  TORCH_CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
              "xpool elastic context state atnagent partition indices must be unique");
  auto channel = AtnAgentControlChannel{};
  channel.mapping_ = std::move(mapping);
  channel.device_index_ = device_index;
  channel.partition_indices_ = std::move(partition_indices);
  return channel;
}

bool AtnAgentControlChannel::captures_complete() const {
  const auto entries = mapping_.partition_entries();
  return std::ranges::all_of(partition_indices_, [&](std::size_t index) {
    return std::atomic_ref{entries[index].initial_backing_ready}.load(std::memory_order_acquire) != 0;
  });
}

void AtnAgentControlChannel::publish_device_memory(std::uint64_t total_bytes, std::uint64_t free_bytes) {
  TORCH_CHECK(total_bytes != 0 && free_bytes <= total_bytes,
              "xpool elastic context state device-memory observation is invalid");
  auto &entry = mapping_.device_entries()[device_index_];
  const auto published_total = std::atomic_ref{entry.device_total_bytes}.load(std::memory_order_acquire);
  if (published_total != 0) {
    TORCH_CHECK(published_total == total_bytes &&
                    std::atomic_ref{entry.device_free_bytes}.load(std::memory_order_relaxed) == free_bytes,
                "xpool elastic context state device-memory observation was already published with different values");
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
  TORCH_CHECK(dp_group_count != 0 && dp_group_count <= layout.device_count && layout.device_count % dp_group_count == 0,
              "xpool elastic context state instance DP group count is invalid");
  TORCH_CHECK(group_index < layout.group_count && partition_index < layout.partition_count,
              "xpool elastic context state instance slot index is out of range");
  const auto groups = mapping.group_entries();
  const auto instance_index = partition_index / layout.device_count;
  auto group_row_begin = group_index;
  while (group_row_begin != 0 && groups[group_row_begin - 1].first_partition / layout.device_count == instance_index) {
    --group_row_begin;
  }
  TORCH_CHECK(groups[group_index].first_partition / layout.device_count == instance_index &&
                  groups[group_row_begin].first_partition == instance_index * layout.device_count &&
                  group_row_begin + dp_group_count <= layout.group_count,
              "xpool elastic context state group directory does not describe the instance row");
  const auto tp_size = layout.device_count / dp_group_count;
  for (auto offset = std::size_t{0}; offset < dp_group_count; ++offset) {
    const auto &group = groups[group_row_begin + offset];
    TORCH_CHECK(group.first_partition == instance_index * layout.device_count + offset * tp_size &&
                    group.partition_count == tp_size,
                "xpool elastic context state group directory differs from instance TP/DP geometry");
  }
  const auto &group = groups[group_index];
  TORCH_CHECK(partition_index >= group.first_partition &&
                  partition_index < group.first_partition + group.partition_count,
              "xpool elastic context state partition slot is outside its Capacity Group");
  auto channel = InstanceControlChannel{};
  channel.mapping_ = std::move(mapping);
  channel.group_index_ = group_index;
  channel.group_row_begin_ = group_row_begin;
  channel.dp_group_count_ = dp_group_count;
  channel.partition_index_ = partition_index;
  return channel;
}

void InstanceControlChannel::publish_initial_backing(std::uint64_t backed_bytes) {
  auto &entry = mapping_.partition_entries()[partition_index_];
  if (std::atomic_ref{entry.initial_backing_ready}.load(std::memory_order_acquire) != 0) {
    TORCH_CHECK(std::atomic_ref{entry.initial_backing_bytes}.load(std::memory_order_relaxed) == backed_bytes,
                "xpool elastic context state initial backing is immutable");
    return;
  }
  std::atomic_ref{entry.initial_backing_bytes}.store(backed_bytes, std::memory_order_relaxed);
  std::atomic_ref{entry.initial_backing_ready}.store(1, std::memory_order_release);
}

std::optional<std::vector<std::uint64_t>> InstanceControlChannel::service_ceiling() const {
  auto &entry = mapping_.group_entries()[group_index_];
  if (std::atomic_ref{entry.ceiling_ready}.load(std::memory_order_acquire) == 0) {
    return std::nullopt;
  }
  auto ceiling = std::vector<std::uint64_t>{};
  for (auto &partition : mapping_.partition_entries().subspan(entry.first_partition, entry.partition_count)) {
    ceiling.push_back(std::atomic_ref{partition.service_ceiling_bytes}.load(std::memory_order_relaxed));
  }
  return ceiling;
}

std::vector<std::optional<QuotaCommand>> InstanceControlChannel::read_commands() const {
  auto commands = std::vector<std::optional<QuotaCommand>>{};
  for (auto offset = std::size_t{0}; offset < dp_group_count_; ++offset) {
    commands.push_back(read_command(mapping_, group_row_begin_ + offset));
  }
  return commands;
}

void InstanceControlChannel::publish_completion(QuotaCompletion completion) {
  TORCH_CHECK(completion.sequence != 0, "xpool elastic context state completion sequence is invalid");
  auto &entry = mapping_.partition_entries()[partition_index_];
  const auto previous = read_completion(entry);
  if (previous && previous->sequence == completion.sequence) {
    TORCH_CHECK(*previous == completion, "xpool elastic context state completion mutates an existing sequence");
    return;
  }
  const auto command = read_command(mapping_, group_index_);
  TORCH_CHECK(command && command->sequence == completion.sequence &&
                  (!previous || previous->sequence < completion.sequence),
              "xpool elastic context state completion does not match the current command");
  const auto revision = begin_publication(entry.completion_revision);
  std::atomic_ref{entry.completion_sequence}.store(completion.sequence, std::memory_order_seq_cst);
  std::atomic_ref{entry.completion_backed_bytes}.store(completion.backed_bytes, std::memory_order_seq_cst);
  std::atomic_ref{entry.completion_revision}.store(revision, std::memory_order_seq_cst);
}

void InstanceControlChannel::publish_demand(QuotaDemand demand) {
  auto &entry = mapping_.group_entries()[group_index_];
  TORCH_CHECK(partition_index_ == entry.first_partition,
              "xpool elastic context state demand publisher must be the TP leader");
  TORCH_CHECK(demand.evaluated_sequence != 0 &&
                  demand.required_bytes.has_value() == demand.deadline_monotonic_ns.has_value(),
              "xpool elastic context state demand presence is invalid");
  auto partitions = mapping_.partition_entries().subspan(entry.first_partition, entry.partition_count);
  TORCH_CHECK(!demand.required_bytes || demand.required_bytes->size() == partitions.size(),
              "xpool elastic context state demand vector extent is invalid");
  TORCH_CHECK(std::atomic_ref{entry.demand_evaluated_sequence}.load(std::memory_order_seq_cst) <=
                  demand.evaluated_sequence,
              "xpool elastic context state demand evaluated sequence regressed");
  const auto revision = begin_publication(entry.demand_revision);
  std::atomic_ref{entry.demand_evaluated_sequence}.store(demand.evaluated_sequence, std::memory_order_seq_cst);
  std::atomic_ref{entry.demand_present}.store(demand.required_bytes ? 1 : 0, std::memory_order_seq_cst);
  std::atomic_ref{entry.demand_deadline_monotonic_ns}.store(demand.deadline_monotonic_ns.value_or(0),
                                                            std::memory_order_seq_cst);
  for (auto index = std::size_t{0}; index < partitions.size(); ++index) {
    std::atomic_ref{partitions[index].required_bytes}.store(demand.required_bytes ? (*demand.required_bytes)[index] : 0,
                                                            std::memory_order_seq_cst);
  }
  std::atomic_ref{entry.demand_revision}.store(revision, std::memory_order_seq_cst);
}

void InstanceControlChannel::close() { mapping_.close(); }

} // namespace xpool::ecs
