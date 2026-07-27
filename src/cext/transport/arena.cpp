#include <ATen/Functions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/TypeCast.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <xpool/transport/arena.hpp>

namespace xpool::transport {

TransportArena TransportArena::create(c10::DeviceIndex cuda_device,
                                      const TransportArenaLayout &layout) {
  layout.validate();
  c10::cuda::CUDAGuard device_guard(cuda_device);
  auto allocation = static_cast<void *>(nullptr);
  C10_CUDA_CHECK(cudaMalloc(&allocation, layout.header.total_bytes));
  auto *arena = static_cast<std::uint8_t *>(allocation);
  try {
    C10_CUDA_CHECK(cudaMemset(arena, 0, layout.header.total_bytes));
    C10_CUDA_CHECK(cudaMemcpy(arena, &layout, sizeof(layout), cudaMemcpyHostToDevice));
    const auto mailbox = TransportMailbox{
        .status = static_cast<std::uint32_t>(MailboxStatus::Dormant),
        .result_code = xpool::abi::FfnResultCode::ProtocolMismatch,
        .payload_rows = 0,
        .request = {},
    };
    C10_CUDA_CHECK(cudaMemcpy(arena + layout.mailbox_offset, &mailbox, sizeof(mailbox),
                              cudaMemcpyHostToDevice));
  } catch (...) {
    (void)cudaFree(arena);
    throw;
  }
  return TransportArena{arena, Kind::Owned, layout};
}

TransportArena TransportArena::from_handle(const TransportArenaHandle &handle) {
  auto mapping = static_cast<void *>(nullptr);
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&mapping, handle.value(), cudaIpcMemLazyEnablePeerAccess));
  auto *arena = static_cast<std::uint8_t *>(mapping);
  TransportArenaLayout layout{};
  try {
    C10_CUDA_CHECK(cudaMemcpy(&layout, arena, sizeof(layout), cudaMemcpyDeviceToHost));
    layout.validate();
  } catch (...) {
    (void)cudaIpcCloseMemHandle(arena);
    throw;
  }
  return TransportArena{arena, Kind::Attached, layout};
}

c10::DeviceIndex TransportArena::cuda_device() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot query an empty transport arena device");
  cudaPointerAttributes attributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, base_));
  return c10::checked_convert<c10::DeviceIndex>(attributes.device, "CUDA pointer device");
}

void TransportArena::destroy() {
  if (base_ == nullptr) {
    return;
  }
  switch (kind_) {
  case Kind::Owned:
    C10_CUDA_CHECK(cudaFree(base_));
    break;
  case Kind::Attached:
    C10_CUDA_CHECK(cudaIpcCloseMemHandle(base_));
    break;
  }
  base_ = nullptr;
  layout_ = {};
}

TransportArenaHandle TransportArena::handle() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot export a CUDA IPC handle for an empty arena");
  TORCH_CHECK(kind_ == Kind::Owned, "xpool can only export handles for owned transport arenas");
  TransportArenaHandle result{};
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&result.value(), base_));
  return result;
}

TransportArenaState TransportArena::state() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot read state from an empty transport arena");
  TransportArenaState result{};
  C10_CUDA_CHECK(cudaMemcpy(&result, base_ + layout_.header.state_offset, sizeof(result),
                            cudaMemcpyDeviceToHost));
  return result;
}

MailboxStatus TransportArena::mailbox_status() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot read mailbox state from an empty transport arena");
  auto status = std::uint32_t{0};
  C10_CUDA_CHECK(cudaMemcpy(&status, base_ + layout_.mailbox_offset +
                                        offsetof(TransportMailbox, status),
                            sizeof(status), cudaMemcpyDeviceToHost));
  TORCH_CHECK(status <= static_cast<std::uint32_t>(MailboxStatus::Closed),
              "xpool transport mailbox contains an invalid status value");
  return static_cast<MailboxStatus>(status);
}

TransportTraceSnapshot TransportArena::read_trace() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot read trace from an empty transport arena");
  TransportTraceSnapshot snapshot{};
  if (layout_.trace.capacity == 0) {
    return snapshot;
  }
  const auto arena_state = state();
  snapshot.sequence = arena_state.trace.sequence;
  snapshot.dropped = arena_state.trace.dropped;
  snapshot.records.resize(std::min<std::size_t>(arena_state.trace.sequence, layout_.trace.capacity));
  C10_CUDA_CHECK(cudaMemcpy(snapshot.records.data(), base_ + layout_.trace.records_offset,
                            snapshot.records.size() * sizeof(TransportTraceRecord),
                            cudaMemcpyDeviceToHost));
  return snapshot;
}

} // namespace xpool::transport
