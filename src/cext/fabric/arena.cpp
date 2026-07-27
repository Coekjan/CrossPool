#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nvshmem.h>

#include <xpool/abort.hpp>
#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/layout.hpp>
#include <xpool/fabric/scheduler.hpp>
#include <xpool/fabric/trace.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/wait.hpp>

namespace xpool::fabric {

FabricArena FabricArena::create(const FabricArenaLayout &layout, std::span<const FabricModelLayout> models,
                                std::span<const FabricLayerLayout> layers,
                                const FfnSchedulerPolicy &scheduler_policy) {
  layout.validate();
  TORCH_CHECK(models.size() == layout.model_count,
              "xpool Fabric model table size differs from arena layout");
  TORCH_CHECK(layers.size() == layout.layer_count,
              "xpool Fabric layer table size differs from arena layout");

  auto expected_layer_begin = std::size_t{0};
  auto expected_decode_offset = std::size_t{0};
  auto maximum_prefill_capacity = std::size_t{0};
  auto model_topologies = std::vector<FabricTraceModelTopology>{};
  model_topologies.reserve(models.size());
  for (const auto &model : models) {
    model.validate();
    TORCH_CHECK(model.layer_begin == expected_layer_begin,
                "xpool Fabric model layer ranges do not form the canonical partition");
    TORCH_CHECK(model.decode_payload_offset == expected_decode_offset,
                "xpool Fabric model Decode payload ranges do not form the canonical partition");
    TORCH_CHECK(xpool::utils::checked::prod(model.atn_tp_size, model.atn_dp_size) == layout.atnagent_count,
                "xpool Fabric model topology does not cover every AtnAgent");
    expected_layer_begin = xpool::utils::checked::sum(expected_layer_begin, model.layer_count);
    expected_decode_offset =
        xpool::utils::checked::sum(expected_decode_offset, model.decode_payload_capacity_bytes);
    maximum_prefill_capacity = std::max(maximum_prefill_capacity, model.prefill_payload_capacity_bytes);
    model_topologies.push_back(
        FabricTraceModelTopology{.atn_tp_size = model.atn_tp_size, .atn_dp_size = model.atn_dp_size});

    auto layer_ids = std::unordered_set<std::size_t>{};
    layer_ids.reserve(model.layer_count);
    for (auto layer_index = model.layer_begin; layer_index < model.layer_begin + model.layer_count; ++layer_index) {
      layers[layer_index].validate();
      TORCH_CHECK(layer_ids.insert(layers[layer_index].layer_id).second,
                  "xpool Fabric model contains duplicate layer ids");
    }
  }
  TORCH_CHECK(expected_layer_begin == layout.layer_count,
              "xpool Fabric model layer ranges do not consume the complete layer table");
  TORCH_CHECK(expected_decode_offset ==
                  layout.model_output_payloads_offset - layout.model_input_payloads_offset,
              "xpool Fabric model Decode ranges do not consume the model payload region");
  TORCH_CHECK(maximum_prefill_capacity == layout.executor_payload_capacity_bytes,
              "xpool Fabric Executor capacity is not the maximum model Prefill capacity");

  const auto scheduler = FfnScheduler::from(scheduler_policy);
  const auto total_bytes = layout.header.total_bytes;
  auto *allocation = static_cast<std::uint8_t *>(
      nvshmem_align(xpool::arena::kAllocationAlignment, total_bytes));
  TORCH_CHECK(allocation != nullptr, "xpool failed to allocate the symmetric Fabric arena");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(allocation) %
                      xpool::arena::kAllocationAlignment ==
                  0,
              "xpool NVSHMEM Fabric arena does not satisfy its required alignment");
  try {
    C10_CUDA_CHECK(cudaMemset(allocation, 0, total_bytes));
    C10_CUDA_CHECK(cudaMemcpy(allocation, &layout, sizeof(layout), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(allocation + layout.model_layouts_offset, models.data(), models.size_bytes(),
                              cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(allocation + layout.layer_layouts_offset, layers.data(), layers.size_bytes(),
                              cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(allocation + layout.scheduler_offset, &scheduler, sizeof(scheduler),
                              cudaMemcpyHostToDevice));
  } catch (...) {
    nvshmem_free(allocation);
    throw;
  }
  return FabricArena{allocation, layout, std::move(model_topologies)};
}

FabricArenaState FabricArena::state() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot read state from an empty Fabric arena");
  auto state = FabricArenaState{};
  C10_CUDA_CHECK(cudaMemcpy(&state, base_ + layout_.header.state_offset, sizeof(state), cudaMemcpyDeviceToHost));
  return state;
}

void FabricArena::wait_until_resident_ready(
    const xpool::utils::device::OwnedCudaStream &resident_stream,
    std::chrono::steady_clock::duration timeout) const {
  TORCH_CHECK(base_ != nullptr,
              "xpool cannot await readiness on an empty Fabric arena");
  TORCH_CHECK(resident_stream,
              "xpool FfnAgent Resident readiness requires a live stream");

  auto host_word = at::empty(
      {}, at::TensorOptions{}
              .dtype(at::kInt)
              .device(at::kCPU)
              .pinned_memory(true));
  auto copy_stream = xpool::utils::device::OwnedCudaStream::create();
  const auto *readiness = base_ + layout_.header.state_offset +
                          offsetof(FabricArenaState,
                                   ffnagent_resident_ready);
  auto read_readiness = [&] {
    C10_CUDA_CHECK(cudaMemcpyAsync(
        host_word.data_ptr(), readiness, sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, copy_stream.get()));
    C10_CUDA_CHECK(cudaStreamSynchronize(copy_stream.get()));
    const auto value =
        static_cast<std::uint32_t>(*host_word.data_ptr<std::int32_t>());
    TORCH_CHECK(
        value <= 1,
        "xpool FfnAgent Resident published an invalid readiness value");
    return value == 1;
  };
  const auto result = xpool::utils::wait::until(
      std::chrono::steady_clock::now() + timeout, read_readiness,
      [&] { return resident_stream.query(); }, std::chrono::milliseconds{1});
  copy_stream.destroy();

  switch (result) {
  case xpool::utils::wait::Result::Ready:
    return;
  case xpool::utils::wait::Result::Cancelled:
    TORCH_CHECK(
        false,
        "xpool FfnAgent Resident completed before publishing readiness");
  case xpool::utils::wait::Result::TimedOut:
    TORCH_CHECK(
        false,
        "xpool FfnAgent Resident startup exceeded the bounded deadline");
  }
  xpool::abort();
}

void FabricArena::request_shutdown(const xpool::utils::device::OwnedCudaStream &stream) const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot request shutdown on an empty Fabric arena");
  auto *shutdown = reinterpret_cast<std::uint32_t *>(base_ + layout_.header.state_offset +
                                                     offsetof(FabricArenaState, shutdown));
  stream.write_value(shutdown, 1);
}

FabricTraceSnapshot FabricArena::read_trace() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot read trace from an empty Fabric arena");
  auto snapshot = FabricTraceSnapshot{
      .pe = nvshmem_my_pe(),
      .atnagent_count = layout_.atnagent_count,
      .ffnagent_count = layout_.ffnagent_count,
      .model_topologies = model_topologies_,
      .sequence = 0,
      .dropped = 0,
      .records = {},
  };
  if (layout_.trace.capacity == 0) {
    return snapshot;
  }
  const auto arena_state = state();
  snapshot.sequence = arena_state.trace.sequence;
  snapshot.dropped = arena_state.trace.dropped;
  snapshot.records.resize(std::min<std::size_t>(arena_state.trace.sequence, layout_.trace.capacity));
  C10_CUDA_CHECK(cudaMemcpy(snapshot.records.data(), base_ + layout_.trace.records_offset,
                            snapshot.records.size() * sizeof(FabricTraceRecord), cudaMemcpyDeviceToHost));
  return snapshot;
}

void FabricArena::destroy() {
  if (base_ == nullptr) {
    return;
  }
  nvshmem_free(base_);
  base_ = nullptr;
  layout_ = {};
  model_topologies_.clear();
}

} // namespace xpool::fabric
