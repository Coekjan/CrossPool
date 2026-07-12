#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <xpool/debug/loopback.hpp>
#include <xpool/debug/options.hpp>
#include <xpool/instance.hpp>
#include <xpool/transport.hpp>

namespace xpool::instance {

namespace {

struct InstanceTransportArenaState {
  xpool::transport::TransportArenaHandleHex handle;
  xpool::transport::TransportArena arena;
  xpool::transport::TransportArenaLayout layout;
  std::mutex use_mutex;

  InstanceTransportArenaState(xpool::transport::TransportArenaHandleHex handle,
                              xpool::transport::TransportArena arena,
                              xpool::transport::TransportArenaLayout layout)
      : handle(std::move(handle)), arena(arena), layout(layout) {}

  InstanceTransportArenaState(const InstanceTransportArenaState &) = delete;
  InstanceTransportArenaState &
  operator=(const InstanceTransportArenaState &) = delete;

  InstanceTransportArenaState(InstanceTransportArenaState &&) = delete;
  InstanceTransportArenaState &
  operator=(InstanceTransportArenaState &&) = delete;
};

using InstanceTransportArenaKey = std::pair<std::int64_t, std::int64_t>;

std::mutex g_instance_arena_state_mutex;
std::map<InstanceTransportArenaKey, InstanceTransportArenaState>
    g_instance_arena_states;

} // namespace

void attach_transport_arena(
    std::int64_t instance_index, std::int64_t rank,
    const xpool::transport::TransportArenaHandleHex &handle) {
  TORCH_CHECK(instance_index >= 0,
              "xpool transport arena requires a non-negative instance index");
  TORCH_CHECK(rank >= 0,
              "xpool transport arena requires a non-negative instance rank");
  xpool::transport::TransportArena arena =
      xpool::transport::TransportArena::from_handle_hex(handle);
  try {
    xpool::transport::TransportArenaLayout layout = arena.layout();
    std::lock_guard<std::mutex> lock(g_instance_arena_state_mutex);
    const InstanceTransportArenaKey key = {instance_index, rank};
    auto [iter, inserted] =
        g_instance_arena_states.try_emplace(key, handle, arena, layout);
    if (!inserted) {
      const bool same_handle = iter->second.handle == handle;
      arena.destroy();
      TORCH_CHECK(
          same_handle,
          "xpool instance transport arena is already attached with different "
          "arena; detach the existing arena before attaching a new one");
      return;
    }
  } catch (...) {
    arena.destroy();
    throw;
  }
}

void detach_transport_arena(std::int64_t instance_index, std::int64_t rank) {
  std::lock_guard<std::mutex> lock(g_instance_arena_state_mutex);
  auto iter = g_instance_arena_states.find({instance_index, rank});
  if (iter == g_instance_arena_states.end()) {
    return;
  }
  std::unique_lock<std::mutex> arena_use_lock(iter->second.use_mutex);
  cudaPointerAttributes attributes{};
  C10_CUDA_CHECK(
      cudaPointerGetAttributes(&attributes, iter->second.arena.base));
  c10::cuda::CUDAGuard device_guard(attributes.device);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  iter->second.arena.destroy();
  arena_use_lock.unlock();
  g_instance_arena_states.erase(iter);
}

std::uint32_t transport_error_snapshot(std::int64_t instance_index,
                                       std::int64_t rank) {
  std::unique_lock<std::mutex> state_lock(g_instance_arena_state_mutex);
  auto iter = g_instance_arena_states.find({instance_index, rank});
  TORCH_CHECK(iter != g_instance_arena_states.end(),
              "xpool cannot read transport error for an unattached arena");
  InstanceTransportArenaState &arena_state = iter->second;
  std::unique_lock<std::mutex> arena_use_lock(arena_state.use_mutex);
  state_lock.unlock();
  cudaPointerAttributes attributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, arena_state.arena.base));
  c10::cuda::CUDAGuard device_guard(attributes.device);
  return arena_state.arena.error_code_snapshot();
}

namespace {

at::Tensor
ffn_shim_req_send(const at::Tensor &hidden_states,
                  const xpool::abi::FfnRequestMetadata &request_metadata,
                  const xpool::abi::FfnTensorMetadata &tensor_metadata,
                  std::int64_t rank,
                  const std::optional<at::Tensor> &global_num_tokens_gpu) {
  std::unique_lock<std::mutex> state_lock(g_instance_arena_state_mutex);
  const InstanceTransportArenaKey key = {request_metadata.instance_index, rank};
  auto iter = g_instance_arena_states.find(key);
  TORCH_CHECK(iter != g_instance_arena_states.end(),
              "xpool ffn_shim has no attached instance transport arena for "
              "instance_index=",
              request_metadata.instance_index, ", rank=", rank,
              "; publish daemon-brokered transport arena before invoking the "
              "production shim");
  InstanceTransportArenaState &arena_state = iter->second;
  std::unique_lock<std::mutex> arena_use_lock(arena_state.use_mutex);
  state_lock.unlock();
  const xpool::transport::TransportArenaLayout &layout = arena_state.layout;
  layout.validate(tensor_metadata, request_metadata);

  at::Tensor output = at::empty_like(hidden_states);
  c10::cuda::CUDAGuard device_guard(hidden_states.device());
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(hidden_states.device().index());
  const xpool::transport::TransportRequest transport_request{
      stream, arena_state.arena, hidden_states,   global_num_tokens_gpu,
      output, request_metadata,  tensor_metadata,
  };
  xpool::transport::launch_instance_transport_kernel(transport_request);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor
ffn_shim_cuda(const at::Tensor &hidden_states,
              const xpool::abi::FfnRequestMetadata &request_metadata,
              const xpool::abi::FfnTensorMetadata &tensor_metadata,
              std::int64_t rank,
              const std::optional<at::Tensor> &global_num_tokens_gpu) {
  if (xpool::debug::options().loopback_site() ==
      xpool::abi::DebugLoopbackSite::kInstance) {
    at::Tensor output = at::empty_like(hidden_states);
    xpool::debug::launch_loopback_rotation(output, hidden_states);
    return output;
  }
  return ffn_shim_req_send(hidden_states, request_metadata, tensor_metadata,
                           rank, global_num_tokens_gpu);
}

at::Tensor ffn_shim_meta(const at::Tensor &hidden_states,
                         const xpool::abi::FfnRequestMetadata &request_metadata,
                         const xpool::abi::FfnTensorMetadata &tensor_metadata,
                         std::int64_t rank) {
  if (xpool::debug::options().loopback_site() ==
      xpool::abi::DebugLoopbackSite::kInstance) {
    return at::empty_like(hidden_states);
  }
  const xpool::transport::TransportArenaLayout layout = [&] {
    std::lock_guard<std::mutex> lock(g_instance_arena_state_mutex);
    const InstanceTransportArenaKey key = {request_metadata.instance_index,
                                           rank};
    auto iter = g_instance_arena_states.find(key);
    TORCH_CHECK(iter != g_instance_arena_states.end(),
                "xpool ffn_shim has no attached instance transport arena for "
                "instance_index=",
                request_metadata.instance_index, ", rank=", rank,
                "; publish daemon-brokered transport arena before invoking the "
                "production shim");
    return iter->second.layout;
  }();
  layout.validate(tensor_metadata, request_metadata);
  return at::empty_like(hidden_states);
}

} // namespace

at::Tensor ffn_shim(const at::Tensor &hidden_states,
                    const xpool::abi::FfnRequestMetadata &request_metadata,
                    std::int64_t rank,
                    const std::optional<at::Tensor> &global_num_tokens_gpu) {
  request_metadata.validate(rank, global_num_tokens_gpu);
  const xpool::abi::FfnTensorMetadata tensor_metadata =
      xpool::abi::FfnTensorMetadata::parse(hidden_states);
  if (hidden_states.is_cuda()) {
    return ffn_shim_cuda(hidden_states, request_metadata, tensor_metadata, rank,
                         global_num_tokens_gpu);
  }
  if (hidden_states.is_meta()) {
    return ffn_shim_meta(hidden_states, request_metadata, tensor_metadata,
                         rank);
  }
  TORCH_CHECK(false, "xpool ffn_shim expects a CUDA or Meta tensor");
}

} // namespace xpool::instance
