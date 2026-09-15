#include <xpool/transport/arena.hpp>

#include <cstddef>
#include <cstdint>

#include <ATen/Functions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/TypeCast.h>
#include <cuda_runtime_api.h>

namespace xpool::transport {

Arena Arena::create(c10::DeviceIndex cuda_device, const ArenaLayout &layout) {
  layout.validate();
  c10::cuda::CUDAGuard device_guard(cuda_device);
  auto allocation = static_cast<void *>(nullptr);
  C10_CUDA_CHECK(cudaMalloc(&allocation, layout.header.total_bytes));
  auto *arena = static_cast<std::uint8_t *>(allocation);
  try {
    C10_CUDA_CHECK(cudaMemset(arena, 0, layout.header.total_bytes));
    C10_CUDA_CHECK(cudaMemcpy(arena, &layout, sizeof(layout), cudaMemcpyHostToDevice));
    const auto mailbox = Mailbox{
        .status = MailboxStatus::Dormant,
        .result_code = xpool::ffn::ResultCode::ProtocolMismatch,
        .payload_rows = 0,
        .request = {},
    };
    C10_CUDA_CHECK(cudaMemcpy(arena + layout.mailbox_offset, &mailbox, sizeof(mailbox), cudaMemcpyHostToDevice));
  } catch (...) {
    (void)cudaFree(arena);
    throw;
  }
  return Arena{arena, Kind::Owned, layout};
}

Arena Arena::from_handle(const ArenaHandle &handle) {
  auto mapping = static_cast<void *>(nullptr);
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&mapping, handle.value(), cudaIpcMemLazyEnablePeerAccess));
  auto *arena = static_cast<std::uint8_t *>(mapping);
  ArenaLayout layout{};
  try {
    C10_CUDA_CHECK(cudaMemcpy(&layout, arena, sizeof(layout), cudaMemcpyDeviceToHost));
    layout.validate();
  } catch (...) {
    (void)cudaIpcCloseMemHandle(arena);
    throw;
  }
  return Arena{arena, Kind::Attached, layout};
}

c10::DeviceIndex Arena::cuda_device() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot query an empty transport arena device");
  cudaPointerAttributes attributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, base_));
  return c10::checked_convert<c10::DeviceIndex>(attributes.device, "CUDA pointer device");
}

void Arena::destroy() {
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

ArenaHandle Arena::handle() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot export a CUDA IPC handle for an empty arena");
  TORCH_CHECK(kind_ == Kind::Owned, "xpool can only export handles for owned transport arenas");
  ArenaHandle result{};
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&result.value(), base_));
  return result;
}

ArenaState Arena::state() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot read state from an empty transport arena");
  ArenaState result{};
  C10_CUDA_CHECK(cudaMemcpy(&result, base_ + layout_.header.state_offset, sizeof(result), cudaMemcpyDeviceToHost));
  return result;
}

MailboxStatus Arena::mailbox_status() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot read mailbox state from an empty transport arena");
  auto status = std::uint32_t{0};
  C10_CUDA_CHECK(cudaMemcpy(&status, base_ + layout_.mailbox_offset + offsetof(Mailbox, status), sizeof(status),
                            cudaMemcpyDeviceToHost));
  TORCH_CHECK(status <= static_cast<std::uint32_t>(MailboxStatus::Closed),
              "xpool transport mailbox contains an invalid status value");
  return static_cast<MailboxStatus>(status);
}

} // namespace xpool::transport
