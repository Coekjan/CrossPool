#include <xpool/transport/instance.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <cuda_runtime_api.h>

#include <xpool/hooks.hpp>
#include <xpool/transport/hooks.hpp>
#include <xpool/utils/wait.hpp>

namespace xpool::transport {

namespace {

constexpr auto kEndpointStartupTimeout = std::chrono::seconds{60};
constexpr auto kEndpointStartupPollInterval = std::chrono::milliseconds{1};

} // namespace

void InstanceRankRuntime::Attachment::attach(std::size_t instance_index, std::size_t rank, const ArenaHandle &handle) {
  if (*this) {
    TORCH_CHECK(matches(instance_index, rank, handle),
                "xpool instance transport arena is already attached with a different identity or handle");
    return;
  }

  auto arena = Arena::from_handle(handle);
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
        TORCH_CHECK(status == MailboxStatus::Dormant,
                    "xpool Transport endpoint entered an invalid state before first use");
        return false;
      },
      kEndpointStartupPollInterval);
  TORCH_CHECK(result == xpool::utils::wait::Status::Ready,
              "xpool Transport endpoint startup exceeded the bounded deadline");
  xpool::hooks::TransportEndpointOpenPostEvent::hooks({.cuda_device = arena.cuda_device(),
                                                       .arena = arena.view(),
                                                       .layout = arena.layout(),
                                                       .site = xpool::hooks::TransportEndpointSite::Instance});
  handle_ = handle;
  arena_ = std::move(arena);
}

void InstanceRankRuntime::Attachment::detach() {
  if (!*this) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(arena_.cuda_device());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  xpool::hooks::TransportEndpointClosePreEvent::hooks({.cuda_device = arena_.cuda_device(),
                                                       .arena = arena_.view(),
                                                       .layout = arena_.layout(),
                                                       .site = xpool::hooks::TransportEndpointSite::Instance});
  arena_.destroy();
  handle_ = {};
}

xpool::ffn::ResultCode InstanceRankRuntime::Attachment::read_generation_failure() const {
  c10::cuda::CUDAGuard device_guard(arena_.cuda_device());
  return arena_.read_generation_failure();
}

void InstanceRankRuntime::Attachment::submit(const at::Tensor &hidden_states,
                                             const std::optional<at::Tensor> &dp_rank_payload_rows,
                                             const at::Tensor &output,
                                             const xpool::transport::RequestMetadata &request_metadata) const {
  const auto &layout = arena_.layout();
  const auto payload_rows = c10::checked_convert<std::size_t>(hidden_states.size(0), "hidden-state rows");
  const auto dp_rank_payload_rows_present =
      dp_rank_payload_rows.has_value() && dp_rank_payload_rows->defined() && dp_rank_payload_rows->numel() != 0;
  TORCH_CHECK(request_metadata.valid(), "xpool FFN shim received invalid request metadata");
  TORCH_CHECK(payload_rows != 0 && payload_rows <= layout.payload_row_capacity,
              "xpool FFN request row count exceeds Transport capacity");
  if (layout.atn_dp_size == 1) {
    TORCH_CHECK(request_metadata.dp_row_layout == xpool::ffn::DpRowLayout::None && !dp_rank_payload_rows_present,
                "xpool DP-one request must use NONE without a per-rank row vector");
  } else {
    TORCH_CHECK(request_metadata.dp_row_layout == xpool::ffn::DpRowLayout::UniformByRank ||
                    request_metadata.dp_row_layout == xpool::ffn::DpRowLayout::PackedByRank,
                "xpool DP request requires UNIFORM_BY_RANK or PACKED_BY_RANK");
    TORCH_CHECK(dp_rank_payload_rows_present, "xpool DP request requires one physical row value per attention DP rank");
  }
  TORCH_CHECK(c10::checked_convert<std::size_t>(hidden_states.size(1), "hidden-state width") == layout.hidden_size,
              "xpool FFN shim hidden size does not match the attached Transport arena");
  TORCH_CHECK(hidden_states.scalar_type() == layout.payload_dtype,
              "xpool FFN shim dtype does not match the attached Transport arena");
  TORCH_CHECK(hidden_states.get_device() == arena_.cuda_device(),
              "xpool FFN shim hidden states must be on the attached Transport arena device");

  if (dp_rank_payload_rows_present) {
    const auto &rank_payload_rows = *dp_rank_payload_rows;
    TORCH_CHECK(c10::checked_convert<std::size_t>(rank_payload_rows.numel(), "DP-rank row vector length") ==
                    layout.atn_dp_size,
                "xpool FFN shim DP-rank payload rows must have one entry per attention DP rank");
  }

  c10::cuda::CUDAGuard device_guard(hidden_states.device());
  const auto stream = at::cuda::getCurrentCUDAStream(hidden_states.device().index());
  const Request request{
      stream, arena_.view(), hidden_states, dp_rank_payload_rows, output, request_metadata,
  };
  launch_request_kernel(request);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void InstanceRankRuntime::attach_arena(std::size_t instance_index, std::size_t rank, const ArenaHandle &handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  attachment_.attach(instance_index, rank, handle);
}

void InstanceRankRuntime::detach_arena() {
  std::lock_guard<std::mutex> lock(mutex_);
  attachment_.detach();
}

xpool::ffn::ResultCode InstanceRankRuntime::read_generation_failure() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(attachment_, "xpool transport failure read requires an attached arena");
  return attachment_.read_generation_failure();
}

void InstanceRankRuntime::submit(const at::Tensor &hidden_states, const std::optional<at::Tensor> &dp_rank_payload_rows,
                                 const at::Tensor &output, const xpool::transport::RequestMetadata &request_metadata) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(attachment_, "xpool ffn_shim has no attached instance transport arena");
  attachment_.submit(hidden_states, dp_rank_payload_rows, output, request_metadata);
}

} // namespace xpool::transport
