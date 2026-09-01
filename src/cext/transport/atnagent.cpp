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
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <xpool/ffn.hpp>
#include <xpool/abort.hpp>
#include <xpool/fabric/runtime.hpp>
#include <xpool/hooks.hpp>
#include <xpool/transport/atnagent.hpp>
#include <xpool/transport/hooks.hpp>
#include <xpool/utils/wait.hpp>

namespace xpool::transport {

namespace {

constexpr auto kResidentStartupTimeout = std::chrono::seconds{60};
constexpr auto kResidentStartupPollInterval = std::chrono::milliseconds{1};

} // namespace

AtnAgentRuntime::Resident::Resident(c10::DeviceIndex cuda_device, std::span<const ArenaView> arenas,
                                             xpool::fabric::ArenaView fabric_arena)
    : cuda_device_(cuda_device) {
  const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
  try {
    auto *arena_allocation = static_cast<void *>(nullptr);
    C10_CUDA_CHECK(cudaMalloc(&arena_allocation, arenas.size_bytes()));
    arenas_ = {static_cast<ArenaView *>(arena_allocation), arenas.size()};
    C10_CUDA_CHECK(cudaMemcpy(arenas_.data(), arenas.data(), arenas.size_bytes(), cudaMemcpyHostToDevice));

    auto *state_allocation = static_cast<void *>(nullptr);
    C10_CUDA_CHECK(cudaMalloc(&state_allocation, sizeof(ResidentState)));
    state_ = static_cast<ResidentState *>(state_allocation);
    C10_CUDA_CHECK(cudaMemset(state_, 0, sizeof(ResidentState)));

    // The control stream owns host-to-device lifecycle publication; the
    // resident stream owns the long-running kernel. Keeping them separate lets
    // drain remain asynchronous without ordering shutdown behind the resident.
    control_stream_ = xpool::utils::device::OwnedCudaStream::create();
    resident_stream_ = xpool::utils::device::OwnedCudaStream::create();
    launch_resident_kernel(arenas_, fabric_arena, state_, resident_stream_.get());
  } catch (...) {
    if (!arenas_.empty()) {
      C10_CUDA_IGNORE_ERROR(cudaFree(arenas_.data()));
      arenas_ = {};
    }
    if (state_ != nullptr) {
      C10_CUDA_IGNORE_ERROR(cudaFree(state_));
      state_ = nullptr;
    }
    throw;
  }
}

AtnAgentRuntime::Resident::~Resident() {
  if (!arenas_.empty()) {
    C10_CUDA_IGNORE_ERROR(cudaFree(arenas_.data()));
  }
  if (state_ != nullptr) {
    C10_CUDA_IGNORE_ERROR(cudaFree(state_));
  }
}

bool AtnAgentRuntime::Resident::pending() const {
  if (released()) {
    return false;
  }
  const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
  return !control_stream_.query() || !resident_stream_.query();
}

void AtnAgentRuntime::Resident::request_drain() {
  if (host_drain_requested_) {
    return;
  }
  const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
  auto *drain_requested = reinterpret_cast<std::uint32_t *>(reinterpret_cast<std::uint8_t *>(state_) +
                                                            offsetof(ResidentState, drain_requested));
  control_stream_.write_value(drain_requested, 1);
  host_drain_requested_ = true;
}

void AtnAgentRuntime::Resident::release() {
  if (released()) {
    return;
  }
  const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
  // Device arrays remain valid until both lifecycle publication and resident
  // execution have completed. Stream destruction precedes allocation release.
  control_stream_.destroy();
  resident_stream_.destroy();
  C10_CUDA_CHECK(cudaFree(arenas_.data()));
  arenas_ = {};
  C10_CUDA_CHECK(cudaFree(state_));
  state_ = nullptr;
}

Arena &AtnAgentRuntime::arena(const ArenaHandle &handle) {
  const auto iter = arenas_.find(handle);
  TORCH_CHECK(iter != arenas_.end(), "xpool Transport arena handle is unknown or already destroyed");
  return iter->second;
}

const Arena &AtnAgentRuntime::arena(const ArenaHandle &handle) const {
  const auto iter = arenas_.find(handle);
  TORCH_CHECK(iter != arenas_.end(), "xpool Transport arena handle is unknown or already destroyed");
  return iter->second;
}

std::vector<ArenaView> AtnAgentRuntime::ordered_views() const {
  auto ordered = std::vector<std::reference_wrapper<const Arena>>{};
  ordered.reserve(arenas_.size());
  for (const auto &entry : arenas_) {
    ordered.emplace_back(entry.second);
  }
  std::ranges::sort(ordered, {}, [](const auto &arena) { return arena.get().layout().instance_index; });

  auto views = std::vector<ArenaView>{};
  views.reserve(ordered.size());
  for (const auto &arena : ordered) {
    views.push_back(arena.get().view());
  }
  return views;
}

bool AtnAgentRuntime::generation_failed() const {
  for (const auto &entry : arenas_) {
    const auto code = entry.second.read_generation_failure();
    if (code != xpool::ffn::ResultCode::Ok) {
      return true;
    }
  }
  return false;
}

ArenaHandle AtnAgentRuntime::create_arena(c10::DeviceIndex cuda_device, std::size_t instance_index,
                                                            std::size_t instance_rank, std::size_t payload_row_capacity,
                                                            std::size_t hidden_size, c10::ScalarType payload_dtype,
                                                            std::size_t atn_tp_rank, std::size_t atn_tp_size,
                                                            std::size_t atn_dp_rank, std::size_t atn_dp_size) {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!resident_.has_value(), "xpool cannot create a Transport arena after Resident activation");
  TORCH_CHECK(!cuda_device_.has_value() || *cuda_device_ == cuda_device,
              "xpool AtnAgent Transport arenas must share one CUDA device");
  for (const auto &entry : arenas_) {
    TORCH_CHECK(entry.second.layout().instance_index != instance_index,
                "xpool AtnAgent already owns a Transport arena for this instance index");
  }

  const auto layout = ArenaLayout::create(instance_index, instance_rank, atn_tp_rank, atn_tp_size, atn_dp_rank,
                                                   atn_dp_size, payload_row_capacity, hidden_size, payload_dtype);
  auto arena = Arena::create(cuda_device, layout);
  xpool::hooks::TransportEndpointOpenPostEvent::hooks({.cuda_device = cuda_device,
                                                       .arena = arena.view(),
                                                       .layout = arena.layout(),
                                                       .site = xpool::hooks::TransportEndpointSite::AtnAgent});
  const auto handle = arena.handle();
  TORCH_CHECK(arenas_.try_emplace(handle, std::move(arena)).second,
              "xpool AtnAgent created a duplicate Transport arena handle");
  cuda_device_ = cuda_device;
  return handle;
}

void AtnAgentRuntime::activate() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!resident_.has_value(), "xpool Transport Resident is already active or terminal");
  TORCH_CHECK(cuda_device_.has_value() && !arenas_.empty(), "xpool Transport Resident requires a non-empty arena set");

  const auto fabric_arena = xpool::fabric::Runtime::singleton().arena();
  const auto views = ordered_views();
  resident_.emplace(*cuda_device_, std::span<const ArenaView>{views}, fabric_arena);

  const auto ready = [&] {
    auto ready = true;
    for (const auto &entry : arenas_) {
      const auto status = entry.second.mailbox_status();
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
      std::chrono::steady_clock::now() + kResidentStartupTimeout, ready, [&] { return !resident_->pending(); },
      kResidentStartupPollInterval);
  switch (result) {
  case xpool::utils::wait::Status::Pending:
    break;
  case xpool::utils::wait::Status::Ready:
    return;
  case xpool::utils::wait::Status::Cancelled:
    TORCH_CHECK(false, "xpool Transport Resident completed before publishing every endpoint ready");
  case xpool::utils::wait::Status::TimedOut:
    TORCH_CHECK(false, "xpool Transport Resident startup exceeded the bounded deadline");
  }
  xpool::abort();
}

void AtnAgentRuntime::check_health() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(resident_.has_value() && !resident_->released(),
              "xpool Transport Resident has not been activated or was released");
  if (resident_->pending()) {
    return;
  }
  TORCH_CHECK(resident_->host_drain_requested() || generation_failed(),
              "xpool Transport Resident completed unexpectedly");
}

void AtnAgentRuntime::drain_async() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!arenas_.empty(), "xpool Transport Resident is unavailable after arena destruction");
  TORCH_CHECK(resident_.has_value(), "xpool Transport Resident has not been activated");
  if (resident_->released()) {
    TORCH_CHECK(resident_->host_drain_requested(),
                "xpool Transport Resident was released without a host drain request");
    return;
  }
  resident_->request_drain();
}

bool AtnAgentRuntime::drain_pending() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!arenas_.empty(), "xpool Transport Resident is unavailable after arena destruction");
  TORCH_CHECK(resident_.has_value() && resident_->host_drain_requested(),
              "xpool Transport Resident drain has not been started");
  if (resident_->released()) {
    return false;
  }
  if (resident_->pending()) {
    return true;
  }
  resident_->release();
  return false;
}

void AtnAgentRuntime::destroy_arenas(std::span<const ArenaHandle> handles) {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!resident_.has_value() || resident_->released(),
              "xpool Transport arenas require preactivation rollback or completed Resident drain");

  auto distinct = std::set<ArenaHandle>{};
  for (const auto &handle : handles) {
    TORCH_CHECK(distinct.insert(handle).second, "xpool Transport arena collection contains a duplicate handle");
    static_cast<void>(arena(handle));
  }

  auto failures = std::vector<std::string>{};
  for (const auto &handle : handles) {
    try {
      auto &resolved = arena(handle);
      xpool::hooks::TransportEndpointClosePreEvent::hooks({.cuda_device = *cuda_device_,
                                                           .arena = resolved.view(),
                                                           .layout = resolved.layout(),
                                                           .site = xpool::hooks::TransportEndpointSite::AtnAgent});
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
