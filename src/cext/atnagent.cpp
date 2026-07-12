#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include <xpool/atnagent.hpp>
#include <xpool/debug/options.hpp>
#include <xpool/transport.hpp>
#include <xpool/utils/device.hpp>

namespace xpool::atnagent {

namespace {

struct AtnAgentTransportArenaState {
  xpool::transport::TransportArena arena;
  std::int64_t cuda_device = 0;
  cudaStream_t stream = nullptr;

  AtnAgentTransportArenaState() = default;

  AtnAgentTransportArenaState(xpool::transport::TransportArena arena,
                              std::int64_t cuda_device)
      : arena(arena), cuda_device(cuda_device) {}

  AtnAgentTransportArenaState(const AtnAgentTransportArenaState &) = delete;
  AtnAgentTransportArenaState &
  operator=(const AtnAgentTransportArenaState &) = delete;

  AtnAgentTransportArenaState(AtnAgentTransportArenaState &&other) noexcept
      : arena(std::exchange(other.arena, xpool::transport::TransportArena{})),
        cuda_device(std::exchange(other.cuda_device, 0)),
        stream(std::exchange(other.stream, nullptr)) {}

  AtnAgentTransportArenaState &
  operator=(AtnAgentTransportArenaState &&other) noexcept {
    if (this != &other) {
      arena = std::exchange(other.arena, xpool::transport::TransportArena{});
      cuda_device = std::exchange(other.cuda_device, 0);
      stream = std::exchange(other.stream, nullptr);
    }
    return *this;
  }
};

std::mutex g_atnagent_arena_mutex;
std::unordered_map<xpool::transport::TransportArenaHandleHex,
                   AtnAgentTransportArenaState>
    g_atnagent_arena_states;

xpool::abi::TransportTraceSnapshot
cleanup_atnagent_arena(AtnAgentTransportArenaState &arena_state) {
  c10::cuda::CUDAGuard device_guard(
      xpool::utils::device::cuda_device_index(arena_state.cuda_device));
  xpool::utils::device::ScopedCudaStream shutdown_stream;

  if (arena_state.stream != nullptr) {
    arena_state.arena.request_shutdown(shutdown_stream.get());
    C10_CUDA_CHECK(cudaStreamSynchronize(shutdown_stream.get()));
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds{60};
    while (true) {
      const cudaError_t query = cudaStreamQuery(arena_state.stream);
      if (query == cudaSuccess) {
        break;
      }
      C10_CUDA_CHECK(query == cudaErrorNotReady ? cudaSuccess : query);
      TORCH_CHECK(std::chrono::steady_clock::now() < deadline,
                  "xpool timed out draining the resident transport kernel; "
                  "the CUDA IPC arena remains allocated");
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    C10_CUDA_CHECK(cudaStreamDestroy(arena_state.stream));
    arena_state.stream = nullptr;
  }
  shutdown_stream.reset();
  xpool::abi::TransportTraceSnapshot snapshot =
      arena_state.arena.trace_snapshot();
  arena_state.arena.destroy();
  return snapshot;
}

AtnAgentTransportArenaState
take_atnagent_arena(const xpool::transport::TransportArenaHandleHex &handle) {
  std::lock_guard<std::mutex> lock(g_atnagent_arena_mutex);
  auto iter = g_atnagent_arena_states.find(handle);
  if (iter == g_atnagent_arena_states.end()) {
    return {};
  }
  AtnAgentTransportArenaState arena_state = std::move(iter->second);
  g_atnagent_arena_states.erase(iter);
  return arena_state;
}

} // namespace

xpool::transport::TransportArenaHandleHex create_transport_arena(
    std::int64_t cuda_device, std::int64_t max_tokens, std::int64_t hidden_size,
    std::int64_t element_size_bytes, std::int64_t atn_dp_size) {
  xpool::transport::TransportArenaLayout layout{
      xpool::transport::kTransportArenaSlotCountDefault, max_tokens,
      hidden_size, element_size_bytes, atn_dp_size};
  xpool::transport::TransportArena arena =
      xpool::transport::TransportArena::create(cuda_device, layout);
  const xpool::transport::TransportArenaHandleHex handle = arena.handle().hex();

  bool inserted = false;
  {
    std::lock_guard<std::mutex> lock(g_atnagent_arena_mutex);
    inserted =
        g_atnagent_arena_states.try_emplace(handle, arena, cuda_device).second;
  }
  if (!inserted) {
    arena.destroy();
    TORCH_CHECK(false, "xpool atnagent created a duplicate transport arena "
                       "handle");
  }

  return handle;
}

xpool::abi::TransportTraceSnapshot destroy_transport_arena(
    const xpool::transport::TransportArenaHandleHex &handle) {
  AtnAgentTransportArenaState arena_state = take_atnagent_arena(handle);
  TORCH_CHECK(arena_state.arena.base != nullptr,
              "xpool cannot destroy an unknown or already destroyed CUDA IPC "
              "arena handle");
  return cleanup_atnagent_arena(arena_state);
}

void launch_transport_kernel(
    const xpool::transport::TransportArenaHandleHex &handle) {
  std::lock_guard<std::mutex> lock(g_atnagent_arena_mutex);
  auto iter = g_atnagent_arena_states.find(handle);
  TORCH_CHECK(iter != g_atnagent_arena_states.end(),
              "xpool cannot launch transport kernel for an unknown CUDA IPC "
              "arena handle");
  AtnAgentTransportArenaState &arena_state = iter->second;
  if (arena_state.stream != nullptr) {
    return;
  }

  c10::cuda::CUDAGuard device_guard(
      xpool::utils::device::cuda_device_index(arena_state.cuda_device));
  xpool::utils::device::ScopedCudaStream stream;
  xpool::transport::launch_atnagent_transport_kernel(arena_state.arena,
                                                     stream.get());
  arena_state.stream = stream.release();
}

} // namespace xpool::atnagent
