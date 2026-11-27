#include <xpool/fabric/arena.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <nvshmem.h>

#include <xpool/arena.hpp>
#include <xpool/fabric/layout.hpp>

namespace xpool::fabric {

void Arena::initialize(const ArenaLayout &layout, std::span<const InstanceEntry> instances,
                       std::span<const LayerEntry> layers, std::span<const int> atnagent_pes,
                       std::span<const int> ffnagent_pes) {
  const auto total_bytes = layout.header.total_bytes;
  base_ = static_cast<std::uint8_t *>(nvshmem_align(xpool::arena::kAllocationAlignment, total_bytes));
  TORCH_CHECK(base_ != nullptr, "xpool failed to allocate the symmetric Fabric arena");
  layout_ = layout;
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(base_) % xpool::arena::kAllocationAlignment == 0,
              "xpool NVSHMEM Fabric arena does not satisfy its required alignment");
  C10_CUDA_CHECK(cudaMemset(base_, 0, total_bytes));
  C10_CUDA_CHECK(cudaMemcpy(base_, &layout, sizeof(layout), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(base_ + layout.instance_entries_offset_bytes, instances.data(), instances.size_bytes(),
                            cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(base_ + layout.layer_entries_offset_bytes, layers.data(), layers.size_bytes(),
                            cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(base_ + layout.atnagent_pes_offset_bytes, atnagent_pes.data(), atnagent_pes.size_bytes(),
                            cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(base_ + layout.ffnagent_pes_offset_bytes, ffnagent_pes.data(), ffnagent_pes.size_bytes(),
                            cudaMemcpyHostToDevice));
}

ArenaState Arena::state() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot read state from an empty Fabric arena");
  auto state = ArenaState{};
  C10_CUDA_CHECK(cudaMemcpy(&state, base_ + layout_.header.state_offset, sizeof(state), cudaMemcpyDeviceToHost));
  return state;
}

void Arena::request_shutdown(const xpool::utils::cuda::OwnedCudaStream &stream) const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot request shutdown on an empty Fabric arena");
  auto *shutdown =
      reinterpret_cast<std::uint32_t *>(base_ + layout_.header.state_offset + offsetof(ArenaState, shutdown));
  stream.write_value(shutdown, 1);
}

void Arena::destroy() {
  if (base_ == nullptr) {
    return;
  }
  nvshmem_free(base_);
  base_ = nullptr;
  layout_ = {};
}

} // namespace xpool::fabric
