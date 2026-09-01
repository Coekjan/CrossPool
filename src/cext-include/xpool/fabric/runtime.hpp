#pragma once

/// \file xpool/fabric/runtime.hpp
/// \brief Host lifecycle boundary for one generation-scoped NVSHMEM Fabric.

#include <c10/core/Device.h>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/ffnagent.hpp>
#include <xpool/fabric/layout.hpp>
#include <xpool/fabric/module.hpp>
#include <xpool/fabric/projection.hpp>
#include <xpool/fabric/scheduler.hpp>
#include <xpool/ffnagent/runtime.hpp>
#include <xpool/utils/device.hpp>

namespace xpool::fabric {

/// Create one opaque NVSHMEM bootstrap unique id as lowercase hexadecimal.
/// \throws c10::Error when NVSHMEM cannot create an id.
Uid create_uid();

/// Process-lifetime owner of one native Fabric participant lifecycle.
/// Public transitions enforce their required phase and runtime role. Invalid
/// transitions and CUDA or NVSHMEM failures surface as c10::Error.
class Runtime {
public:
  /// Return the sole process-lifetime Fabric runtime.
  static Runtime &singleton() {
    static Runtime runtime;
    return runtime;
  }

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  Runtime(Runtime &&) = delete;
  Runtime &operator=(Runtime &&) = delete;

  /// Join one NVSHMEM world and allocate its symmetric arena.
  /// \pre projection passed intrinsic validation at construction.
  void join(c10::DeviceIndex cuda_device, const ArenaProjection &projection, int pe);

  /// Launch and boundedly await the FfnAgent Resident startup publication.
  /// \pre This process joined as an FfnAgent and has not begun drain.
  void activate_ffnagent();

  /// Install one complete FfnAgent execution after Fabric join.
  void install_ffnagent_execution(const xpool::ffnagent::ExecutionProjection &projection);

  /// Reject premature completion or CUDA failure of the active Resident.
  void check_ffnagent_health() const;

  /// Return the participant's current typed symmetric arena view.
  ArenaView arena() const;

  /// Begin terminal local Fabric drain without host synchronization.
  /// Repeated calls during or after completed drain are no-ops.
  void drain_async();

  /// Poll whether local Fabric device work remains in terminal drain.
  bool drain_pending();

  /// Return the locally published canonical Fabric failure when present.
  std::optional<Failure> failure() const;

  /// Collectively release symmetric storage and finalize this local PE.
  /// \pre Local drain completed and every generation participant enters this
  /// collective exactly once.
  void finalize();

private:
  enum class Phase {
    Empty,
    Joined,
    Draining,
    Drained,
    Closed,
  };

  Runtime() = default;
  ~Runtime() = default;

  mutable std::mutex mutex_;
  // Joined, Draining, and Drained own one CUDA device, Projection, PE identity,
  // Fabric arena, module registration, and join-time drain stream. Closed owns
  // none of those resources.
  Phase phase_ = Phase::Empty;
  std::optional<c10::DeviceIndex> cuda_device_;
  std::optional<ArenaProjection> projection_;
  std::optional<int> pe_;
  Arena arena_;
  ModuleRegistration module_registration_;
  FfnAgentControl ffnagent_control_;
  xpool::utils::device::OwnedCudaStream drain_stream_;
  xpool::utils::device::OwnedCudaStream resident_stream_;
  xpool::ffnagent::ExecutionRuntime ffn_execution_runtime_;
};

} // namespace xpool::fabric
