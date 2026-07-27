#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

#include <xpool/transport/instance.hpp>
#include <xpool/utils/wait.hpp>

namespace xpool::transport {

namespace {

constexpr auto kEndpointStartupTimeout = std::chrono::seconds{60};
constexpr auto kEndpointStartupPollInterval = std::chrono::milliseconds{1};

xpool::abi::TensorDType tensor_dtype(c10::ScalarType scalar_type) {
  switch (scalar_type) {
  case c10::ScalarType::BFloat16:
    return xpool::abi::TensorDType{xpool::abi::TensorDType::Bf16};
  case c10::ScalarType::Half:
    return xpool::abi::TensorDType{xpool::abi::TensorDType::Fp16};
  case c10::ScalarType::Float:
    return xpool::abi::TensorDType{xpool::abi::TensorDType::Fp32};
  default:
    TORCH_CHECK(false, "xpool transport supports only float32, float16, and bfloat16 tensors");
  }
}

} // namespace

void InstanceTransportRuntime::Attachment::attach(std::size_t instance_index, std::size_t rank,
                                                  const TransportArenaHandle &handle) {
  if (*this) {
    TORCH_CHECK(matches(instance_index, rank, handle),
                "xpool instance transport arena is already attached with a different identity or handle");
    return;
  }

  auto arena = TransportArena::from_handle(handle);
  const auto &layout = arena.layout();
  TORCH_CHECK(layout.instance_index == instance_index && layout.instance_rank == rank,
              "xpool transport arena identity does not match the attaching instance process");
  const auto result = xpool::utils::wait::until(
      std::chrono::steady_clock::now() + kEndpointStartupTimeout,
      [&] {
        const auto status = arena.mailbox_status();
        if (status == MailboxStatus::Idle) {
          return true;
        }
        TORCH_CHECK(
            status == MailboxStatus::Dormant,
            "xpool Transport endpoint entered an invalid state before first use");
        return false;
      },
      kEndpointStartupPollInterval);
  TORCH_CHECK(
      result == xpool::utils::wait::Result::Ready,
      "xpool Transport endpoint startup exceeded the bounded deadline");
  handle_ = handle;
  arena_ = std::move(arena);
}

void InstanceTransportRuntime::Attachment::detach() {
  if (!*this) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(arena_.cuda_device());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  arena_.destroy();
  handle_ = {};
}

xpool::abi::FfnResultCode
InstanceTransportRuntime::Attachment::read_generation_failure() const {
  c10::cuda::CUDAGuard device_guard(arena_.cuda_device());
  return arena_.read_generation_failure();
}

at::Tensor InstanceTransportRuntime::Attachment::submit(
    const at::Tensor &hidden_states, const std::optional<at::Tensor> &global_num_tokens_gpu,
    const xpool::transport::FfnRequestMetadata &request_metadata) const {
  const auto &layout = arena_.layout();
  const auto payload_rows = c10::checked_convert<std::size_t>(hidden_states.size(0), "hidden-state rows");
  const auto token_counts_present = global_num_tokens_gpu.has_value() && global_num_tokens_gpu->defined() &&
                                    global_num_tokens_gpu->numel() != 0;
  request_metadata.validate(layout, payload_rows, token_counts_present);
  TORCH_CHECK(c10::checked_convert<std::size_t>(hidden_states.size(1), "hidden-state width") == layout.hidden_size,
              "xpool FFN shim hidden size does not match the attached Transport arena");
  TORCH_CHECK(tensor_dtype(hidden_states.scalar_type()).value() == layout.dtype,
              "xpool FFN shim dtype does not match the attached Transport arena");
  TORCH_CHECK(hidden_states.get_device() == arena_.cuda_device(),
              "xpool FFN shim hidden states must be on the attached Transport arena device");

  if (token_counts_present) {
    const auto &token_counts = *global_num_tokens_gpu;
    TORCH_CHECK(token_counts.is_cuda() && token_counts.is_contiguous() && token_counts.dim() == 1,
                "xpool FFN shim DP token counts must be a contiguous 1D CUDA tensor");
    TORCH_CHECK(token_counts.scalar_type() == at::kInt || token_counts.scalar_type() == at::kLong,
                "xpool FFN shim DP token counts must be int32 or int64");
    TORCH_CHECK(token_counts.get_device() == hidden_states.get_device(),
                "xpool FFN shim DP token counts must be on the hidden-state device");
    TORCH_CHECK(c10::checked_convert<std::size_t>(token_counts.numel(), "DP token count length") ==
                    layout.atn_dp_size,
                "xpool FFN shim DP token counts must have one entry per attention DP rank");
  }

  auto output = at::empty_like(hidden_states);
  TORCH_CHECK(hidden_states.is_cuda(), "xpool transport submit expects a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(hidden_states.device());
  const auto stream = at::cuda::getCurrentCUDAStream(hidden_states.device().index());
  const TransportRequest request{
      stream,
      arena_.view(),
      hidden_states,
      global_num_tokens_gpu,
      output,
      request_metadata,
  };
  launch_request_kernel(request);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void InstanceTransportRuntime::attach_arena(std::size_t instance_index, std::size_t rank,
                                            const TransportArenaHandle &handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  attachment_.attach(instance_index, rank, handle);
}

void InstanceTransportRuntime::detach_arena() {
  std::lock_guard<std::mutex> lock(mutex_);
  attachment_.detach();
}

xpool::abi::FfnResultCode InstanceTransportRuntime::read_generation_failure() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(attachment_, "xpool transport failure read requires an attached arena");
  return attachment_.read_generation_failure();
}

at::Tensor InstanceTransportRuntime::submit(const at::Tensor &hidden_states,
                                            const std::optional<at::Tensor> &global_num_tokens_gpu,
                                            const xpool::transport::FfnRequestMetadata &request_metadata) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(attachment_, "xpool ffn_shim has no attached instance transport arena");
  return attachment_.submit(hidden_states, global_num_tokens_gpu, request_metadata);
}

} // namespace xpool::transport
