#pragma once

/// \file xpool/fabric/ffnagent.hpp
/// \brief Host owner and Device view for per-FfnAgent Fabric control storage.

#include <cstddef>
#include <cstdint>

#include <xpool/fabric/scheduler.hpp>

namespace xpool::fabric {

/// Non-owning Device addresses for one FfnAgent Fabric control allocation.
struct FfnAgentControlView {
  /// Process-local resident activation count.
  std::uint32_t *activation_count = nullptr;
  /// Fabric Coordinator Scheduler, or null on an ordinary FfnAgent.
  Scheduler *coordinator_scheduler = nullptr;
};

/// Explicit owner of process-local FfnAgent control storage.
/// Normal lifecycle calls destroy() so CUDA failures are visible; the
/// destructor performs best-effort fallback cleanup only.
class FfnAgentControl {
public:
  FfnAgentControl() = default;
  ~FfnAgentControl();

  FfnAgentControl(const FfnAgentControl &) = delete;
  FfnAgentControl &operator=(const FfnAgentControl &) = delete;
  /// Transfer ownership and leave other empty.
  FfnAgentControl(FfnAgentControl &&other) noexcept;
  /// Transfer ownership into an empty destination.
  FfnAgentControl &operator=(FfnAgentControl &&other);

  /// Allocate and initialize control storage for one FfnAgent PE.
  /// \throws c10::Error for invalid geometry or CUDA allocation/copy failure.
  static FfnAgentControl create(bool is_coordinator, const SchedulerPolicy &scheduler_policy,
                                std::size_t instance_count, std::size_t executor_lane_count);

  /// Return whether this owner retains an allocation.
  explicit operator bool() const noexcept { return allocation_ != nullptr; }
  /// Return non-owning device addresses valid for this owner's lifetime.
  FfnAgentControlView view() const { return view_; }
  /// Release the allocation; repeated calls are no-ops.
  /// \throws c10::Error when CUDA cannot release a live allocation.
  void destroy();

private:
  std::uint8_t *allocation_ = nullptr;
  FfnAgentControlView view_{};
};

/// Return the exact process-local FfnAgent control allocation size.
/// \throws c10::Error when instance_count is zero or size arithmetic overflows.
std::size_t ffnagent_control_allocation_bytes(bool is_coordinator, std::size_t instance_count);

} // namespace xpool::fabric
