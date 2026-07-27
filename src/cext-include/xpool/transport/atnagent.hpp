#pragma once

/// \file xpool/transport/atnagent.hpp
/// \brief Process-wide AtnAgent Transport Resident and arena lifecycle.

#include <c10/core/Device.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include <xpool/fabric/arena.hpp>
#include <xpool/transport/arena.hpp>
#include <xpool/utils/device.hpp>

namespace xpool::transport {

/// Launch one cooperative Transport Resident over an immutable arena array.
/// \param arenas Device array ordered by TransportArenaLayout::instance_index.
/// \param arena_count Positive number of views and cooperative grid blocks.
/// \param fabric_arena Joined Fabric arena, or empty for AtnAgent loopback.
/// \param state Process-local device control state shared by every block.
/// \param stream Nonblocking stream that owns the Resident launch.
/// \pre Cooperative launch is supported and the complete grid is concurrently
/// resident; capability and occupancy failures occur before launch.
void launch_transport_resident_kernel(const TransportArenaView *arenas,
                                      std::size_t arena_count,
                                      xpool::fabric::FabricArenaView fabric_arena,
                                      TransportResidentState *state,
                                      cudaStream_t stream);

/// Process-lifetime owner of all Transport arenas and one AtnAgent Resident.
///
/// Arena creation and preactivation rollback remain incremental. Activation
/// freezes the complete arena set and launches exactly one cooperative block
/// per arena. Health and drain are process-wide and never accept handles.
class AtnAgentTransportRuntime {
public:
  /// Return the sole process-lifetime AtnAgent Transport runtime.
  /// \return AtnAgent Transport lifecycle owner for this process.
  static AtnAgentTransportRuntime &singleton() {
    static AtnAgentTransportRuntime runtime;
    return runtime;
  }

  AtnAgentTransportRuntime(const AtnAgentTransportRuntime &) = delete;
  AtnAgentTransportRuntime &operator=(const AtnAgentTransportRuntime &) = delete;
  AtnAgentTransportRuntime(AtnAgentTransportRuntime &&) = delete;
  AtnAgentTransportRuntime &operator=(AtnAgentTransportRuntime &&) = delete;

  /// Allocate and register one AtnAgent-owned Transport arena.
  /// \param cuda_device CUDA device shared by this runtime's arena set.
  /// \param instance_index Configuration-order model identity.
  /// \param instance_rank Rank-local process identity.
  /// \param max_tokens Maximum physical rows accepted by one request.
  /// \param hidden_size Hidden-state columns accepted by one request.
  /// \param dtype Exact hidden-state dtype accepted by this arena.
  /// \param atn_tp_rank Attention tensor-parallel coordinate.
  /// \param atn_tp_size Attention tensor-parallel width.
  /// \param atn_dp_rank Attention data-parallel coordinate.
  /// \param atn_dp_size Attention data-parallel width.
  /// \return Lowercase CUDA IPC handle retained until destroy_arenas().
  TransportArenaHandle create_arena(c10::DeviceIndex cuda_device,
                                    std::size_t instance_index,
                                    std::size_t instance_rank,
                                    std::size_t max_tokens,
                                    std::size_t hidden_size,
                                    xpool::abi::TensorDType dtype,
                                    std::size_t atn_tp_rank,
                                    std::size_t atn_tp_size,
                                    std::size_t atn_dp_rank,
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
  /// \return True while the Resident or its control write remains pending;
  /// false after completed drain.
  bool drain_pending();

  /// Read one drained or preactivation arena's Transport trace.
  /// \param handle Handle returned by create_arena().
  /// \return Snapshot, or no value when observation is disabled.
  std::optional<TransportTraceSnapshot> read_trace(const TransportArenaHandle &handle) const;

  /// Destroy known arenas during preactivation rollback or after Resident drain.
  /// \param handles Distinct handles returned by create_arena().
  /// \throws c10::Error after attempting every requested cleanup when any
  /// release fails.
  void destroy_arenas(std::span<const TransportArenaHandle> handles);

private:
  class Resident {
  public:
    Resident(c10::DeviceIndex cuda_device,
             std::span<const TransportArenaView> arenas,
             xpool::fabric::FabricArenaView fabric_arena);
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
    TransportArenaView *arenas_ = nullptr;
    TransportResidentState *state_ = nullptr;
    xpool::utils::device::OwnedCudaStream control_stream_;
    xpool::utils::device::OwnedCudaStream resident_stream_;
    bool host_drain_requested_ = false;
  };

  AtnAgentTransportRuntime() = default;
  ~AtnAgentTransportRuntime() = default;

  bool generation_failed() const;
  std::vector<TransportArenaView> ordered_views() const;
  TransportArena &arena(const TransportArenaHandle &handle);
  const TransportArena &arena(const TransportArenaHandle &handle) const;

  mutable std::mutex mutex_;
  std::optional<c10::DeviceIndex> cuda_device_;
  std::map<TransportArenaHandle, TransportArena> arenas_;
  std::optional<Resident> resident_;
};

} // namespace xpool::transport
