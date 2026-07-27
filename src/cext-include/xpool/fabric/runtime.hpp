#pragma once

/// \file xpool/fabric/runtime.hpp
/// \brief Host lifecycle boundary for one generation-scoped NVSHMEM Fabric.

#include <c10/core/Device.h>
#include <nvshmemx.h>

#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

#include <xpool/abi.hpp>
#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/layout.hpp>
#include <xpool/fabric/module.hpp>
#include <xpool/fabric/scheduler.hpp>
#include <xpool/fabric/trace.hpp>
#include <xpool/utils/device.hpp>
#include <xpool/utils/hex.hpp>

namespace xpool::fabric {

/// Opaque NVSHMEM unique id with canonical hexadecimal projection.
using FabricUid = xpool::utils::hex::HexValue<nvshmemx_uniqueid_t>;

/// One ordered FFN layer supplied to native Fabric join.
struct FabricLayerMetadata {
  /// Model decoder layer identifier used for executor binding.
  std::size_t layer_id;
  /// Valid structural executor kind for this layer.
  FfnLayerKind kind;

  /// Compare two layer metadata records.
  /// \return True when layer identity and kind are equal.
  bool operator==(const FabricLayerMetadata &) const = default;
};

/// One model's topology, capacities, and ordered FFN layers at native join.
struct FabricModelMetadata {
  /// Maximum Decode rows reserved in model-owned payload storage.
  std::size_t max_decode_rows;
  /// Maximum Prefill rows reserved in Executor-owned payload storage.
  std::size_t max_prefill_rows;
  /// Hidden-state element type shared by this model's Fabric payloads.
  xpool::abi::TensorDType dtype;
  /// Hidden-state columns in every invocation for this model.
  std::size_t hidden_size;
  /// Tensor-parallel AtnAgent width for this model.
  std::size_t atn_tp_size;
  /// Data-parallel AtnAgent width for this model.
  std::size_t atn_dp_size;
  /// Decoder FFN layers in canonical execution order.
  std::vector<FabricLayerMetadata> layers;

  /// Compare two model metadata records.
  /// \return True when all topology, capacity, and layer fields are equal.
  bool operator==(const FabricModelMetadata &) const = default;
};

/// Complete semantic input for one process joining an NVSHMEM Fabric world.
struct FabricJoinMetadata {
  /// Opaque NVSHMEM bootstrap UID shared by every participant.
  FabricUid uid;
  /// This process's contiguous NVSHMEM PE index.
  int pe;
  /// Number of AtnAgent PEs in the canonical prefix.
  std::size_t atnagent_count;
  /// Number of FfnAgent PEs in the canonical suffix.
  std::size_t ffnagent_count;
  /// Number of independently admitted distributed FFN Executors.
  std::size_t executor_count;
  /// Immutable policy used to construct the arena Scheduler.
  FfnSchedulerPolicy scheduler_policy;
  /// Models in configuration order.
  std::vector<FabricModelMetadata> models;

  /// Return the total participant count.
  /// \return Sum of AtnAgent and FfnAgent PE counts.
  std::size_t pe_count() const { return atnagent_count + ffnagent_count; }

  /// Return the first FfnAgent PE and sole Coordinator owner.
  /// \return NVSHMEM PE index of the Coordinator.
  int coordinator_pe() const { return static_cast<int>(atnagent_count); }

  /// Validate UID, topology, capacities, policy, and ordered layers.
  void validate() const;

  /// Compare two complete Fabric join contracts.
  /// \return True when every semantic join input is equal.
  bool operator==(const FabricJoinMetadata &) const = default;
};

/// Create one opaque NVSHMEM bootstrap unique id as lowercase hexadecimal.
/// \return Bootstrap UID suitable for every participant in one generation.
FabricUid create_uid();

/// Process-lifetime owner of one native Fabric participant lifecycle.
class FabricRuntime {
public:
  /// Return the sole process-lifetime Fabric runtime.
  /// \return Process-global Fabric lifecycle owner.
  static FabricRuntime &singleton() {
    static FabricRuntime runtime;
    return runtime;
  }

  FabricRuntime(const FabricRuntime &) = delete;
  FabricRuntime &operator=(const FabricRuntime &) = delete;
  FabricRuntime(FabricRuntime &&) = delete;
  FabricRuntime &operator=(FabricRuntime &&) = delete;

  /// Join one NVSHMEM world and allocate its symmetric arena.
  /// \param cuda_device CUDA device used by this participant.
  /// \param metadata Complete generation-scoped join contract.
  void join(c10::DeviceIndex cuda_device, const FabricJoinMetadata &metadata);

  /// Launch and boundedly await the FfnAgent Resident startup publication.
  void activate();

  /// Reject premature completion or CUDA failure of the active Resident.
  void check_health() const;

  /// Return the participant's current typed symmetric arena view.
  /// \return Empty view before join, otherwise the process-local arena view.
  FabricArenaView arena() const;

  /// Begin terminal local Fabric drain without host synchronization.
  /// Repeated calls during or after completed drain are no-ops.
  void drain_async();

  /// Poll whether local Fabric device work remains in terminal drain.
  /// \return True while the drain command or Resident remains pending; false
  /// after completed drain.
  bool drain_pending();

  /// Return the locally published canonical Fabric failure when present.
  /// \return Retained canonical failure, or no value when none is published.
  std::optional<FabricFailure> failure() const;

  /// Read this drained PE's local Fabric trace when observation is enabled.
  /// \return Trace snapshot, or no value when observation is disabled.
  std::optional<FabricTraceSnapshot> read_trace();

  /// Collectively release symmetric storage and finalize this local PE.
  void shutdown();

private:
  enum class Phase {
    Empty,
    Joined,
    Draining,
    Drained,
    Finalized,
  };

  FabricRuntime() = default;
  ~FabricRuntime() = default;

  mutable std::mutex mutex_;
  Phase phase_ = Phase::Empty;
  std::optional<c10::DeviceIndex> cuda_device_;
  std::optional<FabricJoinMetadata> metadata_;
  FabricArena arena_;
  FabricModuleRegistration module_registration_;
  xpool::utils::device::OwnedCudaStream drain_stream_;
  xpool::utils::device::OwnedCudaStream resident_stream_;
};

} // namespace xpool::fabric
