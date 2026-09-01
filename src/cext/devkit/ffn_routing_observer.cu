#include <xpool/devkit/ffn_routing_observer.hpp>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <ATen/ATen.h>

#include <cooperative_groups.h>
#include <cuda/std/span>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

#include <xpool/arena.hpp>
#include <xpool/debug/options.cuh>
#include <xpool/devkit/adapters.cuh>
#include <xpool/devkit/adapters.hpp>
#include <xpool/fabric/arena.cuh>
#include <xpool/macros.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/layout.hpp>
#include <xpool/utils/trace.cuh>

namespace xpool::devkit::ffn_routing_observer {

namespace {

struct RecordHeader {
  xpool::fabric::InvocationKey key;
  std::size_t layer_ordinal;
  std::size_t executor_lane_index;
  std::uint64_t executor_lease_sequence;
  std::uint32_t rows;
  std::uint32_t effective_topk;
};

struct BufferLayout {
  std::size_t record_capacity;
  std::size_t element_capacity;
  std::size_t state_offset;
  std::size_t headers_offset;
  std::size_t ids_offset;
  std::size_t weights_offset;
  std::size_t total_bytes;

  static BufferLayout create(std::size_t record_capacity, std::size_t element_capacity) {
    TORCH_CHECK(record_capacity != 0, "routing observer record capacity must be positive");
    TORCH_CHECK(element_capacity != 0, "routing observer element capacity must be positive");
    const auto total_elements = xpool::utils::checked::prod(record_capacity, element_capacity);
    const auto plan = xpool::utils::layout::LayoutPlan<4>{
        std::array{
            xpool::utils::layout::LayoutRegionSpec::object<xpool::utils::trace::BufferState>("state"),
            xpool::utils::layout::LayoutRegionSpec::array<RecordHeader>("headers", record_capacity),
            xpool::utils::layout::LayoutRegionSpec::array<std::int32_t>("ids", total_elements),
            xpool::utils::layout::LayoutRegionSpec::array<float>("weights", total_elements),
        },
        xpool::arena::kAllocationAlignment};
    auto index = std::size_t{0};
    const auto &state = plan[index++];
    const auto &headers = plan[index++];
    const auto &ids = plan[index++];
    const auto &weights = plan[index++];
    TORCH_CHECK(index == plan.regions.size(), "xpool FFN Routing Observer region plan is incomplete");
    return BufferLayout{
        .record_capacity = record_capacity,
        .element_capacity = element_capacity,
        .state_offset = state.offset,
        .headers_offset = headers.offset,
        .ids_offset = ids.offset,
        .weights_offset = weights.offset,
        .total_bytes = plan.total_bytes,
    };
  }
};

class BufferView {
public:
  XPOOL_HOST_DEVICE_FN BufferView(std::uint8_t *base, BufferLayout layout)
      : base_(base), layout_(layout) {}

  XPOOL_HOST_DEVICE_FN std::uint8_t *base() const { return base_; }
  XPOOL_HOST_DEVICE_FN const BufferLayout &layout() const { return layout_; }

  XPOOL_HOST_DEVICE_FN xpool::utils::trace::BufferState &state() const {
    return *reinterpret_cast<xpool::utils::trace::BufferState *>(base_ + layout_.state_offset);
  }

  XPOOL_HOST_DEVICE_FN std::int32_t *ids(std::size_t index) const {
    return reinterpret_cast<std::int32_t *>(base_ + layout_.ids_offset) + index * layout_.element_capacity;
  }

  XPOOL_HOST_DEVICE_FN float *weights(std::size_t index) const {
    return reinterpret_cast<float *>(base_ + layout_.weights_offset) + index * layout_.element_capacity;
  }

private:
  std::uint8_t *base_;
  BufferLayout layout_;
};

struct Storage {
  std::uint8_t *buffer = nullptr;
  BufferLayout layout{};
};

static_assert(std::is_trivially_copyable_v<RecordHeader>);
static_assert(std::is_trivially_copyable_v<BufferLayout>);
static_assert(std::is_trivially_copyable_v<BufferView>);
static_assert(std::is_trivially_copyable_v<Storage>);

XPOOL_DEVICE_CONST Storage storage{};

std::mutex state_mutex;
std::uint8_t *buffer = nullptr;
BufferLayout layout{};
bool installed = false;

} // namespace

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionInstallPreEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().ffn_routing_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  TORCH_CHECK(!installed && buffer == nullptr, "xpool Routing Observer is already installed");
  auto element_capacity = std::size_t{0};
  for (const auto &instance : context.fabric_projection.instances) {
    const auto payload_row_capacity =
        std::max(instance.decode_payload_row_capacity, instance.prefill_payload_row_capacity);
    for (const auto &layer : instance.layers) {
      if (layer.kind == xpool::ffn::LayerKind::Moe && !layer.ffnagent_indices.empty() &&
          layer.ffnagent_indices.front() == context.ffnagent_index) {
        element_capacity = std::max(
            element_capacity, xpool::utils::checked::prod(payload_row_capacity, layer.effective_topk));
      }
    }
  }
  if (element_capacity != 0) {
    layout = BufferLayout::create(
        xpool::debug::options().ffn_routing_observer.record_capacity, element_capacity);
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer), layout.total_bytes));
    C10_CUDA_CHECK(cudaMemset(buffer, 0, layout.total_bytes));
  }
  const auto installed_storage = Storage{.buffer = buffer, .layout = layout};
  C10_CUDA_CHECK(cudaMemcpyToSymbol(storage, &installed_storage, sizeof(installed_storage)));
  installed = true;
}

XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricFfnAgentProtocolEvent, DeviceAdapter::observe, context) {
  if (!xpool::debug::options().ffn_routing_observer.enable ||
      context.kind != xpool::hooks::FabricFfnAgentProtocolEvent::Kind::RoutingMetadataPublished ||
      storage.buffer == nullptr) {
    return;
  }

  const auto &execution = *context.execution;
  const auto &layer = context.arena.layer_entry(execution.key.instance_index, execution.layer_ordinal);
  if (execution.payload_rows > storage.layout.element_capacity / layer.effective_topk) {
    return;
  }

  const auto group = cooperative_groups::this_thread_block();
  const auto view = BufferView{storage.buffer, storage.layout};
  const auto metadata = context.arena.routing_metadata(context.executor_lane_index, context.payload_row_capacity);
  const auto live_element_count = execution.payload_rows * layer.effective_topk;
  XPOOL_DEVICE_SHARED std::uint64_t reserved_sequence;
  // One thread reserves and initializes the header before the CTA copies the
  // corresponding routing payload into the same one-based record slot.
  if (group.thread_rank() == 0) {
    const auto headers =
        cuda::std::span{reinterpret_cast<RecordHeader *>(view.base() + view.layout().headers_offset),
                        view.layout().record_capacity};
    const auto records = xpool::utils::trace::BufferView<RecordHeader>{headers, view.state()};
    const auto entry = records.reserve();
    reserved_sequence = entry ? entry->sequence() : 0;
    if (entry) {
      entry->record() = RecordHeader{
          .key = execution.key,
          .layer_ordinal = execution.layer_ordinal,
          .executor_lane_index = context.executor_lane_index,
          .executor_lease_sequence = execution.executor_lease_sequence,
          .rows = static_cast<std::uint32_t>(execution.payload_rows),
          .effective_topk = static_cast<std::uint32_t>(layer.effective_topk),
      };
    }
  }
  group.sync();

  if (reserved_sequence == 0) {
    return;
  }
  const auto record_index = static_cast<std::size_t>(reserved_sequence - 1);
  const auto topk_ids = metadata.topk_ids();
  const auto topk_weights = metadata.topk_weights();
  for (auto index = static_cast<std::size_t>(group.thread_rank()); index < live_element_count;
       index += static_cast<std::size_t>(group.size())) {
    view.ids(record_index)[index] = topk_ids[index];
    view.weights(record_index)[index] = topk_weights[index];
  }
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionFinalizePostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().ffn_routing_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  const auto empty_storage = Storage{};
  C10_CUDA_CHECK(cudaMemcpyToSymbol(storage, &empty_storage, sizeof(empty_storage)));
  if (buffer != nullptr) {
    C10_CUDA_CHECK(cudaFree(buffer));
  }
  buffer = nullptr;
  layout = {};
  installed = false;
}

std::optional<Snapshot> read() {
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  if (!installed) {
    return std::nullopt;
  }
  if (buffer == nullptr) {
    return Snapshot{.sequence = 0, .dropped = 0, .records = {}};
  }
  const auto view = BufferView{buffer, layout};
  auto state = xpool::utils::trace::BufferState{};
  C10_CUDA_CHECK(cudaMemcpy(&state, view.base() + view.layout().state_offset, sizeof(state), cudaMemcpyDeviceToHost));
  TORCH_CHECK(state.dropped <= state.sequence && state.sequence - state.dropped <= view.layout().record_capacity,
              "routing observer counters are inconsistent");
  const auto record_count = static_cast<std::size_t>(state.sequence - state.dropped);
  auto headers = std::vector<RecordHeader>(record_count);
  if (!headers.empty()) {
    C10_CUDA_CHECK(cudaMemcpy(headers.data(), view.base() + view.layout().headers_offset,
                              headers.size() * sizeof(RecordHeader), cudaMemcpyDeviceToHost));
  }

  auto snapshot = Snapshot{.sequence = state.sequence, .dropped = state.dropped, .records = {}};
  snapshot.records.reserve(record_count);
  for (auto index = std::size_t{0}; index < record_count; ++index) {
    const auto &header = headers[index];
    TORCH_CHECK(header.key.valid() && header.rows != 0 && header.effective_topk != 0 &&
                    header.rows <= view.layout().element_capacity / header.effective_topk,
                "routing observer record is invalid");
    const auto rows = static_cast<std::int64_t>(header.rows);
    const auto effective_topk = static_cast<std::int64_t>(header.effective_topk);
    auto ids = at::empty({rows, effective_topk}, at::TensorOptions{}.dtype(at::kInt).device(at::kCPU));
    auto weights = at::empty({rows, effective_topk}, at::TensorOptions{}.dtype(at::kFloat).device(at::kCPU));
    const auto element_count = static_cast<std::size_t>(header.rows) * header.effective_topk;
    C10_CUDA_CHECK(
        cudaMemcpy(ids.data_ptr(), view.ids(index), element_count * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(
        cudaMemcpy(weights.data_ptr(), view.weights(index), element_count * sizeof(float), cudaMemcpyDeviceToHost));
    snapshot.records.push_back(Record{
        .key = header.key,
        .layer_ordinal = header.layer_ordinal,
        .topk_ids = std::move(ids),
        .topk_weights = std::move(weights),
    });
  }
  return snapshot;
}

std::size_t allocation_bytes(std::size_t element_capacity) {
  const auto &observer = xpool::debug::options().ffn_routing_observer;
  if (!observer.enable || element_capacity == 0) {
    return 0;
  }
  return BufferLayout::create(observer.record_capacity, element_capacity).total_bytes;
}

} // namespace xpool::devkit::ffn_routing_observer
