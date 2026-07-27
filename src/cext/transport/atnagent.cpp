#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <span>
#include <sstream>
#include <string>
#include <set>
#include <utility>
#include <vector>

#include <xpool/abort.hpp>
#include <xpool/abi.hpp>
#include <xpool/debug/options.hpp>
#include <xpool/fabric/runtime.hpp>
#include <xpool/transport/atnagent.hpp>
#include <xpool/utils/wait.hpp>

namespace xpool::transport {

namespace {

constexpr auto kResidentStartupTimeout = std::chrono::seconds{60};
constexpr auto kResidentStartupPollInterval = std::chrono::milliseconds{1};

} // namespace

AtnAgentTransportRuntime::Resident::Resident(
    c10::DeviceIndex cuda_device, std::span<const TransportArenaView> arenas,
    xpool::fabric::FabricArenaView fabric_arena)
    : cuda_device_(cuda_device) {
  TORCH_CHECK(!arenas.empty(),
              "xpool Transport Resident requires at least one arena");
  const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
  try {
    auto *arena_allocation = static_cast<void *>(nullptr);
    C10_CUDA_CHECK(
        cudaMalloc(&arena_allocation, arenas.size_bytes()));
    arenas_ = static_cast<TransportArenaView *>(arena_allocation);
    C10_CUDA_CHECK(cudaMemcpy(arenas_, arenas.data(), arenas.size_bytes(),
                              cudaMemcpyHostToDevice));

    auto *state_allocation = static_cast<void *>(nullptr);
    C10_CUDA_CHECK(cudaMalloc(&state_allocation, sizeof(TransportResidentState)));
    state_ = static_cast<TransportResidentState *>(state_allocation);
    C10_CUDA_CHECK(cudaMemset(state_, 0, sizeof(TransportResidentState)));

    // The control stream owns host-to-device lifecycle publication; the
    // resident stream owns the long-running kernel. Keeping them separate lets
    // drain remain asynchronous without ordering shutdown behind the resident.
    control_stream_ = xpool::utils::device::OwnedCudaStream::create();
    resident_stream_ = xpool::utils::device::OwnedCudaStream::create();
    launch_transport_resident_kernel(arenas_, arenas.size(), fabric_arena,
                                     state_, resident_stream_.get());
  } catch (...) {
    if (arenas_ != nullptr) {
      C10_CUDA_IGNORE_ERROR(cudaFree(arenas_));
      arenas_ = nullptr;
    }
    if (state_ != nullptr) {
      C10_CUDA_IGNORE_ERROR(cudaFree(state_));
      state_ = nullptr;
    }
    throw;
  }
}

AtnAgentTransportRuntime::Resident::~Resident() {
  if (arenas_ != nullptr) {
    C10_CUDA_IGNORE_ERROR(cudaFree(arenas_));
  }
  if (state_ != nullptr) {
    C10_CUDA_IGNORE_ERROR(cudaFree(state_));
  }
}

bool AtnAgentTransportRuntime::Resident::pending() const {
  if (released()) {
    return false;
  }
  const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
  return !control_stream_.query() || !resident_stream_.query();
}

void AtnAgentTransportRuntime::Resident::request_drain() {
  TORCH_CHECK(!released(),
              "xpool Transport Resident resources have already been released");
  if (host_drain_requested_) {
    return;
  }
  const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
  auto *drain_requested = reinterpret_cast<std::uint32_t *>(
      reinterpret_cast<std::uint8_t *>(state_) +
      offsetof(TransportResidentState, drain_requested));
  control_stream_.write_value(drain_requested, 1);
  host_drain_requested_ = true;
}

void AtnAgentTransportRuntime::Resident::release() {
  if (released()) {
    return;
  }
  const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
  TORCH_CHECK(control_stream_.query() && resident_stream_.query(),
              "xpool cannot release a pending Transport Resident");
  // Device arrays remain valid until both lifecycle publication and resident
  // execution have completed. Stream destruction precedes allocation release.
  control_stream_.destroy();
  resident_stream_.destroy();
  C10_CUDA_CHECK(cudaFree(arenas_));
  arenas_ = nullptr;
  C10_CUDA_CHECK(cudaFree(state_));
  state_ = nullptr;
}

TransportArena &AtnAgentTransportRuntime::arena(
    const TransportArenaHandle &handle) {
  const auto iter = arenas_.find(handle);
  TORCH_CHECK(iter != arenas_.end(),
              "xpool Transport arena handle is unknown or already destroyed");
  return iter->second;
}

const TransportArena &AtnAgentTransportRuntime::arena(
    const TransportArenaHandle &handle) const {
  const auto iter = arenas_.find(handle);
  TORCH_CHECK(iter != arenas_.end(),
              "xpool Transport arena handle is unknown or already destroyed");
  return iter->second;
}

std::vector<TransportArenaView> AtnAgentTransportRuntime::ordered_views() const {
  auto ordered = std::vector<std::reference_wrapper<const TransportArena>>{};
  ordered.reserve(arenas_.size());
  for (const auto &[handle, arena] : arenas_) {
    static_cast<void>(handle);
    ordered.emplace_back(arena);
  }
  std::ranges::sort(ordered, {}, [](const auto &arena) {
    return arena.get().layout().instance_index;
  });

  auto views = std::vector<TransportArenaView>{};
  views.reserve(ordered.size());
  for (const auto &arena : ordered) {
    views.push_back(arena.get().view());
  }
  return views;
}

bool AtnAgentTransportRuntime::generation_failed() const {
  for (const auto &[handle, arena] : arenas_) {
    static_cast<void>(handle);
    const auto code = arena.read_generation_failure();
    if (code != xpool::abi::FfnResultCode::Ok) {
      return true;
    }
  }
  return false;
}

TransportArenaHandle AtnAgentTransportRuntime::create_arena(
    c10::DeviceIndex cuda_device, std::size_t instance_index,
    std::size_t instance_rank, std::size_t max_tokens, std::size_t hidden_size,
    xpool::abi::TensorDType dtype, std::size_t atn_tp_rank,
    std::size_t atn_tp_size, std::size_t atn_dp_rank,
    std::size_t atn_dp_size) {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!resident_.has_value(),
              "xpool cannot create a Transport arena after Resident activation");
  TORCH_CHECK(!cuda_device_.has_value() || *cuda_device_ == cuda_device,
              "xpool AtnAgent Transport arenas must share one CUDA device");
  for (const auto &[handle, arena] : arenas_) {
    static_cast<void>(handle);
    TORCH_CHECK(arena.layout().instance_index != instance_index,
                "xpool AtnAgent already owns a Transport arena for this instance index");
  }

  const auto layout = TransportArenaLayout::create(
      instance_index, instance_rank, atn_tp_rank, atn_tp_size, atn_dp_rank,
      atn_dp_size, max_tokens, hidden_size, dtype);
  auto arena = TransportArena::create(cuda_device, layout);
  const auto handle = arena.handle();
  TORCH_CHECK(arenas_.try_emplace(handle, std::move(arena)).second,
              "xpool AtnAgent created a duplicate Transport arena handle");
  cuda_device_ = cuda_device;
  return handle;
}

void AtnAgentTransportRuntime::activate() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!resident_.has_value(),
              "xpool Transport Resident is already active or terminal");
  TORCH_CHECK(cuda_device_.has_value() && !arenas_.empty(),
              "xpool Transport Resident requires a non-empty arena set");

  auto fabric_arena = xpool::fabric::FabricArenaView{};
  const auto loopback = xpool::debug::options().loopback;
  if (!loopback.enabled() || loopback.site != xpool::debug::LoopbackSite::AtnAgent) {
    fabric_arena = xpool::fabric::FabricRuntime::singleton().arena();
  }
  const auto views = ordered_views();
  resident_.emplace(*cuda_device_, std::span<const TransportArenaView>{views},
                    fabric_arena);

  const auto ready = [&] {
    auto ready = true;
    for (const auto &[handle, arena] : arenas_) {
      static_cast<void>(handle);
      const auto status = arena.mailbox_status();
      if (status == MailboxStatus::Dormant) {
        ready = false;
      } else {
        TORCH_CHECK(status == MailboxStatus::Idle,
                    "xpool Transport Resident published an invalid startup mailbox state");
      }
    }
    return ready;
  };
  const auto result = xpool::utils::wait::until(
      std::chrono::steady_clock::now() + kResidentStartupTimeout, ready,
      [&] { return !resident_->pending(); }, kResidentStartupPollInterval);
  switch (result) {
  case xpool::utils::wait::Result::Ready:
    return;
  case xpool::utils::wait::Result::Cancelled:
    TORCH_CHECK(
        false,
        "xpool Transport Resident completed before publishing every endpoint ready");
  case xpool::utils::wait::Result::TimedOut:
    TORCH_CHECK(
        false,
        "xpool Transport Resident startup exceeded the bounded deadline");
  }
  xpool::abort();
}

void AtnAgentTransportRuntime::check_health() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(resident_.has_value() && !resident_->released(),
              "xpool Transport Resident has not been activated or was released");
  if (resident_->pending()) {
    return;
  }
  TORCH_CHECK(resident_->host_drain_requested() || generation_failed(),
              "xpool Transport Resident completed unexpectedly");
}

void AtnAgentTransportRuntime::drain_async() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!arenas_.empty(),
              "xpool Transport Resident is unavailable after arena destruction");
  TORCH_CHECK(resident_.has_value(),
              "xpool Transport Resident has not been activated");
  if (resident_->released()) {
    TORCH_CHECK(resident_->host_drain_requested(),
                "xpool Transport Resident was released without a host drain request");
    return;
  }
  resident_->request_drain();
}

bool AtnAgentTransportRuntime::drain_pending() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!arenas_.empty(),
              "xpool Transport Resident is unavailable after arena destruction");
  TORCH_CHECK(resident_.has_value() && resident_->host_drain_requested(),
              "xpool Transport Resident drain has not been started");
  if (resident_->released()) {
    return false;
  }
  if (resident_->pending()) {
    return true;
  }
  TORCH_CHECK(resident_->host_drain_requested() || generation_failed(),
              "xpool Transport Resident completed without an accepted terminal cause");
  // release() is the single terminal transition: after it returns, trace reads
  // are safe and arena destruction can no longer race the Resident.
  resident_->release();
  return false;
}

std::optional<TransportTraceSnapshot> AtnAgentTransportRuntime::read_trace(
    const TransportArenaHandle &handle) const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(resident_.has_value() && resident_->released(),
              "xpool Transport trace requires completed Resident drain");
  const auto &resolved = arena(handle);
  if (resolved.layout().trace.capacity == 0) {
    return std::nullopt;
  }
  return resolved.read_trace();
}

void AtnAgentTransportRuntime::destroy_arenas(
    std::span<const TransportArenaHandle> handles) {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!resident_.has_value() || resident_->released(),
              "xpool Transport arenas require preactivation rollback or completed Resident drain");

  auto distinct = std::set<TransportArenaHandle>{};
  for (const auto &handle : handles) {
    TORCH_CHECK(distinct.insert(handle).second,
                "xpool Transport arena collection contains a duplicate handle");
    static_cast<void>(arena(handle));
  }

  auto failures = std::vector<std::string>{};
  for (const auto &handle : handles) {
    try {
      auto &resolved = arena(handle);
      resolved.destroy();
      arenas_.erase(handle);
    } catch (const std::exception &error) {
      failures.push_back(handle.encode() + ": " + error.what());
    } catch (...) {
      failures.push_back(handle.encode() + ": unknown native cleanup error");
    }
  }
  if (!resident_.has_value() && arenas_.empty()) {
    cuda_device_.reset();
  }
  if (!failures.empty()) {
    auto message = std::ostringstream{};
    message << "xpool failed to destroy Transport arenas";
    for (const auto &failure : failures) {
      message << "\n- " << failure;
    }
    TORCH_CHECK(false, message.str());
  }
}

} // namespace xpool::transport
