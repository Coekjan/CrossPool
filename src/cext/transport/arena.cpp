
#include <ATen/Functions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

#include <xpool/debug/options.hpp>
#include <xpool/transport.hpp>
#include <xpool/utils/arith.hpp>
#include <xpool/utils/device.hpp>
#include <xpool/utils/layout.hpp>
#include <xpool/utils/queue.hpp>

namespace xpool::transport {

namespace {

struct TransportArenaPlan {
  xpool::utils::layout::LayoutRegion layout_header;
  xpool::utils::layout::LayoutRegion shutdown_flag;
  xpool::utils::layout::LayoutRegion error_code;
  xpool::utils::layout::LayoutRegion free_queue_state;
  xpool::utils::layout::LayoutRegion free_queue_cells;
  xpool::utils::layout::LayoutRegion used_queue_state;
  xpool::utils::layout::LayoutRegion used_queue_cells;
  xpool::utils::layout::LayoutRegion request_descriptors;
  xpool::utils::layout::LayoutRegion result_descriptors;
  xpool::utils::layout::LayoutRegion input_buffers;
  xpool::utils::layout::LayoutRegion output_buffers;
  xpool::utils::layout::LayoutRegion dp_token_counts;
  xpool::utils::layout::LayoutRegion trace_sequence;
  xpool::utils::layout::LayoutRegion trace_dropped;
  xpool::utils::layout::LayoutRegion trace_records;
  std::int64_t total_bytes;

  TransportArenaPlan(std::int64_t slot_count, std::int64_t slot_stride_bytes,
                     std::int64_t dp_token_counts_stride_bytes,
                     std::int64_t trace_capacity) {
    using xpool::utils::layout::LayoutRegionSpec;
    const auto specs = std::to_array<LayoutRegionSpec>({
        LayoutRegionSpec::object<TransportArenaLayout>("layout header"),
        LayoutRegionSpec::object<std::uint32_t>("shutdown flag"),
        LayoutRegionSpec::object<std::uint32_t>("sticky error code"),
        LayoutRegionSpec::object<xpool::utils::queue::RingQueueState>(
            "free queue state"),
        LayoutRegionSpec::array<
            xpool::utils::queue::RingQueueCell<std::uint32_t>>(
            "free queue cells", slot_count),
        LayoutRegionSpec::object<xpool::utils::queue::RingQueueState>(
            "used queue state"),
        LayoutRegionSpec::array<
            xpool::utils::queue::RingQueueCell<std::uint32_t>>(
            "used queue cells", slot_count),
        LayoutRegionSpec::array<xpool::abi::FfnRequestDescriptor>(
            "request descriptors", slot_count),
        LayoutRegionSpec::array<xpool::abi::FfnResultDescriptor>(
            "result descriptors", slot_count),
        LayoutRegionSpec::bytes(
            "input buffers", XPOOL_CHECKED_MUL(slot_count, slot_stride_bytes)),
        LayoutRegionSpec::bytes(
            "output buffers", XPOOL_CHECKED_MUL(slot_count, slot_stride_bytes)),
        LayoutRegionSpec::bytes(
            "DP token counts",
            XPOOL_CHECKED_MUL(slot_count, dp_token_counts_stride_bytes)),
        LayoutRegionSpec::bytes(
            "trace sequence", trace_capacity == 0 ? 0 : sizeof(std::uint64_t)),
        LayoutRegionSpec::bytes(
            "trace dropped", trace_capacity == 0 ? 0 : sizeof(std::uint64_t)),
        LayoutRegionSpec::array<xpool::abi::TransportTraceRecord>(
            "trace records", trace_capacity),
    });
    const xpool::utils::layout::LayoutPlan layout{specs,
                                                  kTransportArenaAlignment};
    std::size_t index = 0;
    layout_header = layout[index++];
    shutdown_flag = layout[index++];
    error_code = layout[index++];
    free_queue_state = layout[index++];
    free_queue_cells = layout[index++];
    used_queue_state = layout[index++];
    used_queue_cells = layout[index++];
    request_descriptors = layout[index++];
    result_descriptors = layout[index++];
    input_buffers = layout[index++];
    output_buffers = layout[index++];
    dp_token_counts = layout[index++];
    trace_sequence = layout[index++];
    trace_dropped = layout[index++];
    trace_records = layout[index++];
    total_bytes = layout.total_bytes;
    TORCH_CHECK(index == layout.regions.size(),
                "xpool transport arena plan did not consume every region");
  }
};

void initialize_transport_queues(std::uint8_t *arena,
                                 const TransportArenaLayout &layout) {
  const auto slot_count = static_cast<std::size_t>(layout.slot_count);
  std::vector<std::uint32_t> free_slot_ids(slot_count);
  std::iota(free_slot_ids.begin(), free_slot_ids.end(), 0U);

  const auto free_image =
      xpool::utils::queue::RingQueueHostImage<std::uint32_t>::full(
          std::span<const std::uint32_t>{free_slot_ids.data(),
                                         free_slot_ids.size()});
  const auto used_image =
      xpool::utils::queue::RingQueueHostImage<std::uint32_t>::empty(slot_count);

  C10_CUDA_CHECK(cudaMemcpy(arena + layout.free_queue.state_offset,
                            &free_image.state, sizeof(free_image.state),
                            cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(arena + layout.used_queue.state_offset,
                            &used_image.state, sizeof(used_image.state),
                            cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      arena + layout.free_queue.cells_offset, free_image.cells.data(),
      free_image.cells.size() *
          sizeof(xpool::utils::queue::RingQueueCell<std::uint32_t>),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      arena + layout.used_queue.cells_offset, used_image.cells.data(),
      used_image.cells.size() *
          sizeof(xpool::utils::queue::RingQueueCell<std::uint32_t>),
      cudaMemcpyHostToDevice));
}

} // namespace

TransportArenaLayout::TransportArenaLayout(std::int64_t slot_count,
                                           std::int64_t max_tokens,
                                           std::int64_t hidden_size,
                                           std::int64_t element_size_bytes,
                                           std::int64_t atn_dp_size) {
  TORCH_CHECK(slot_count > 0,
              "xpool transport arena requires a positive slot count");
  XPOOL_CHECKED_U32(slot_count);
  TORCH_CHECK(max_tokens > 0,
              "xpool transport arena requires a positive max token capacity");
  TORCH_CHECK(hidden_size > 0,
              "xpool transport arena requires a positive hidden size");
  TORCH_CHECK(hidden_size % 2 == 0,
              "xpool transport rotation requires an even hidden size");
  TORCH_CHECK(element_size_bytes == 2 || element_size_bytes == 4,
              "xpool transport arena supports only 2-byte and 4-byte hidden "
              "elements");
  TORCH_CHECK(atn_dp_size > 0,
              "xpool transport arena requires a positive attention DP size");

  const std::int64_t slot_stride_bytes = XPOOL_ALIGN_UP_MUL3(
      max_tokens, hidden_size, element_size_bytes, kTransportArenaAlignment);
  const std::int64_t dp_token_counts_stride_bytes = XPOOL_ALIGN_UP_MUL(
      atn_dp_size, static_cast<std::int64_t>(sizeof(std::uint32_t)),
      kTransportArenaAlignment);
  const std::int64_t trace_capacity =
      xpool::debug::options().enabled(
          xpool::abi::DebugOption::kTransportObserver)
          ? kTransportTraceCapacity
          : 0;
  const TransportArenaPlan plan{slot_count, slot_stride_bytes,
                                dp_token_counts_stride_bytes, trace_capacity};
  this->magic = kTransportArenaLayoutMagic;
  this->arena_bytes = plan.total_bytes;
  this->slot_count = slot_count;
  this->slot_stride_bytes = slot_stride_bytes;
  this->free_queue = xpool::utils::queue::RingQueueLayout{
      plan.free_queue_state.offset,
      plan.free_queue_cells.offset,
  };
  this->used_queue = xpool::utils::queue::RingQueueLayout{
      plan.used_queue_state.offset,
      plan.used_queue_cells.offset,
  };
  this->shutdown_offset = plan.shutdown_flag.offset;
  this->error_code_offset = plan.error_code.offset;
  this->request_base_offset = plan.request_descriptors.offset;
  this->result_base_offset = plan.result_descriptors.offset;
  this->input_base_offset = plan.input_buffers.offset;
  this->output_base_offset = plan.output_buffers.offset;
  this->dp_token_counts_base_offset = plan.dp_token_counts.offset;
  this->dp_token_counts_stride_bytes = dp_token_counts_stride_bytes;
  this->max_tokens = max_tokens;
  this->hidden_size = hidden_size;
  this->atn_dp_size = atn_dp_size;
  this->trace_sequence_offset =
      trace_capacity == 0 ? 0 : plan.trace_sequence.offset;
  this->trace_dropped_offset =
      trace_capacity == 0 ? 0 : plan.trace_dropped.offset;
  this->trace_records_offset =
      trace_capacity == 0 ? 0 : plan.trace_records.offset;
  this->trace_capacity = trace_capacity;
}

void TransportArenaLayout::validate(
    const xpool::abi::FfnTensorMetadata &tensor_metadata,
    const xpool::abi::FfnRequestMetadata &request_metadata) const {
  TORCH_CHECK(tensor_metadata.num_tokens <= max_tokens,
              "xpool transport FFN shim token count exceeds arena capacity");
  TORCH_CHECK(tensor_metadata.hidden_size == hidden_size,
              "xpool transport FFN shim hidden size does not match arena");
  TORCH_CHECK(XPOOL_CHECKED_MUL(
                  static_cast<std::int64_t>(tensor_metadata.num_tokens),
                  static_cast<std::int64_t>(tensor_metadata.hidden_size)) <=
                  slot_stride_bytes / tensor_metadata.dtype.elem_size(),
              "xpool transport FFN shim tensor dtype exceeds arena byte "
              "capacity");
  TORCH_CHECK(request_metadata.atn_dp_size == atn_dp_size,
              "xpool transport FFN shim attention DP size does not match "
              "transport arena");
}

TransportArena TransportArena::create(std::int64_t cuda_device,
                                      const TransportArenaLayout &layout) {
  c10::cuda::CUDAGuard device_guard(
      xpool::utils::device::cuda_device_index(cuda_device));

  std::uint8_t *arena = nullptr;
  C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&arena),
                            static_cast<std::size_t>(layout.arena_bytes)));
  try {
    C10_CUDA_CHECK(
        cudaMemset(arena, 0, static_cast<std::size_t>(layout.arena_bytes)));
    C10_CUDA_CHECK(
        cudaMemcpy(arena, &layout, sizeof(layout), cudaMemcpyHostToDevice));
    initialize_transport_queues(arena, layout);
  } catch (...) {
    (void)cudaFree(arena);
    throw;
  }
  return TransportArena{arena, ReleaseKind::kOwnedAllocation};
}

void TransportArena::destroy() {
  if (base == nullptr) {
    return;
  }
  switch (release_kind) {
  case ReleaseKind::kOwnedAllocation:
    C10_CUDA_CHECK(cudaFree(base));
    break;
  case ReleaseKind::kIpcMapping:
    C10_CUDA_CHECK(cudaIpcCloseMemHandle(base));
    break;
  }
  base = nullptr;
}

std::uint32_t TransportArena::error_code_snapshot() const {
  TORCH_CHECK(base != nullptr,
              "xpool cannot read error state from an empty transport arena");
  const TransportArenaLayout host_layout = layout();
  std::uint32_t value = xpool::abi::FfnResultErrorCode::kOk;
  C10_CUDA_CHECK(cudaMemcpy(&value, base + host_layout.error_code_offset,
                            sizeof(value), cudaMemcpyDeviceToHost));
  return value;
}

xpool::abi::TransportTraceSnapshot TransportArena::trace_snapshot() const {
  const TransportArenaLayout host_layout = layout();
  xpool::abi::TransportTraceSnapshot snapshot{};
  if (host_layout.trace_capacity == 0) {
    return snapshot;
  }
  snapshot.records.resize(static_cast<std::size_t>(host_layout.trace_capacity));
  C10_CUDA_CHECK(cudaMemcpy(&snapshot.sequence,
                            base + host_layout.trace_sequence_offset,
                            sizeof(snapshot.sequence), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(&snapshot.dropped,
                            base + host_layout.trace_dropped_offset,
                            sizeof(snapshot.dropped), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      snapshot.records.data(), base + host_layout.trace_records_offset,
      snapshot.records.size() * sizeof(xpool::abi::TransportTraceRecord),
      cudaMemcpyDeviceToHost));
  return snapshot;
}

} // namespace xpool::transport
