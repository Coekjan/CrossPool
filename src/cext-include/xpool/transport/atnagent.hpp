#pragma once

/// \file xpool/transport/atnagent.hpp
/// \brief Process-wide AtnAgent Transport Resident and arena lifecycle.

#include <cstddef>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include <c10/core/Device.h>
#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <xpool/fabric/arena.hpp>
#include <xpool/transport/arena.hpp>
#include <xpool/utils/device.hpp>

namespace xpool::transport {

/// Launch one cooperative Transport Resident over an immutable arena array.
/// \pre arenas is non-empty and state and stream are non-null.
/// \pre Cooperative launch is supported and the complete grid is concurrently
/// resident; capability and occupancy failures occur before launch.
/// \throws c10::Error when capability, occupancy, or launch checks fail.
void launch_resident_kernel(cuda::std::span<const ArenaView> arenas, xpool::fabric::ArenaView fabric_arena,
                            ResidentState *state, cudaStream_t stream);

/// Process-lifetime owner of all Transport arenas and one AtnAgent Resident.
///
/// Arena creation and preactivation rollback remain incremental. Activation
/// freezes the complete arena set and launches exactly one cooperative block
/// per arena. Health and drain are process-wide and never accept handles.
/// Public operations enforce this lifecycle and surface CUDA failures as
/// c10::Error.
class AtnAgentRuntime {
public:
  /// Return the sole process-lifetime AtnAgent Transport runtime.
  static AtnAgentRuntime &singleton() {
    static AtnAgentRuntime runtime;
    return runtime;
  }

  AtnAgentRuntime(const AtnAgentRuntime &) = delete;
  AtnAgentRuntime &operator=(const AtnAgentRuntime &) = delete;
  AtnAgentRuntime(AtnAgentRuntime &&) = delete;
  AtnAgentRuntime &operator=(AtnAgentRuntime &&) = delete;

  /// Allocate and register one AtnAgent-owned Transport arena.
  /// The returned CUDA IPC handle remains valid until destroy_arenas().
  ArenaHandle create_arena(c10::DeviceIndex cuda_device, std::size_t instance_index, std::size_t instance_rank,
                           std::size_t payload_row_capacity, std::size_t hidden_size, c10::ScalarType payload_dtype,
                           std::size_t atn_tp_rank, std::size_t atn_tp_size, std::size_t atn_dp_rank,
                           std::size_t atn_dp_size);

  /// Freeze the arena set and launch its sole cooperative Resident.
  ///
  /// This method returns only after every mailbox is acquire-observed as Idle.
  void activate();

  /// Reject premature Resident completion or CUDA stream failure.
  ///
  /// Completion caused by a retained canonical generation failure is expected;
  /// completion with neither that fact nor a host drain request is an error.
  void check_health() const;

  /// Publish the process-local Resident drain command without synchronizing.
  /// Repeated calls during or after completed drain are no-ops.
  void drain_async();

  /// Poll process-wide Resident completion and release its device resources.
  /// Returns true while the Resident or its control write remains pending.
  bool drain_pending();

  /// Destroy known arenas during preactivation rollback or after Resident drain.
  /// \throws c10::Error after attempting every requested cleanup when any
  /// release fails.
  void destroy_arenas(std::span<const ArenaHandle> handles);

private:
  class Resident {
  public:
    Resident(c10::DeviceIndex cuda_device, std::span<const ArenaView> arenas, xpool::fabric::ArenaView fabric_arena);
    ~Resident();

    Resident(const Resident &) = delete;
    Resident &operator=(const Resident &) = delete;
    Resident(Resident &&) = delete;
    Resident &operator=(Resident &&) = delete;

    bool pending() const;
    bool host_drain_requested() const { return host_drain_requested_; }
    bool released() const { return state_ == nullptr; }
    void request_drain();
    void release();

  private:
    c10::DeviceIndex cuda_device_;
    cuda::std::span<ArenaView> arenas_{};
    ResidentState *state_ = nullptr;
    xpool::utils::device::OwnedCudaStream control_stream_;
    xpool::utils::device::OwnedCudaStream resident_stream_;
    bool host_drain_requested_ = false;
  };

  AtnAgentRuntime() = default;
  ~AtnAgentRuntime() = default;

  bool generation_failed() const;
  std::vector<ArenaView> ordered_views() const;
  Arena &arena(const ArenaHandle &handle);
  const Arena &arena(const ArenaHandle &handle) const;

  mutable std::mutex mutex_;
  std::optional<c10::DeviceIndex> cuda_device_;
  std::map<ArenaHandle, Arena> arenas_;
  std::optional<Resident> resident_;
};

} // namespace xpool::transport
