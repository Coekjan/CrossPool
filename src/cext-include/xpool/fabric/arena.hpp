#pragma once

/// \file xpool/fabric/arena.hpp
/// \brief Fabric symmetric-arena state, typed view, and explicit owner.

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <c10/util/Exception.h>

#include <xpool/abort.hpp>
#include <xpool/fabric/layout.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/fabric/trace.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/device.hpp>

#if defined(__CUDACC__)
#include <xpool/atomic.cuh>
#endif

namespace xpool::fabric {

class FfnScheduler;
class FfnSchedulerEntry;
class FfnSchedulerPolicy;

/// Mutable device state local to one PE's symmetric arena allocation.
struct FabricArenaState {
  /// Monotonic FfnAgent Resident startup publication; unused on AtnAgent PEs.
  std::uint32_t ffnagent_resident_ready;
  /// Monotonic PE-local command requesting Resident drain.
  std::uint32_t shutdown;
  /// First-writer-wins canonical invocation failure.
  FabricFailure failure;
  /// Allocation counters for this PE's local Fabric trace buffer.
  xpool::trace::BufferState trace;
};

/// Non-owning typed address view over one process-local symmetric arena.
struct FabricArenaView {
  /// Construct an empty arena view.
  XPOOL_HOST_DEVICE_FN constexpr FabricArenaView() = default;

  /// Construct a view over one process-local symmetric arena.
  /// \param base Process-local address returned by the symmetric allocation.
  XPOOL_HOST_DEVICE_FN explicit constexpr FabricArenaView(std::uint8_t *base) : base_(base) {}

  /// Return whether no arena is bound.
  /// \return True when this view has no process-local arena address.
  XPOOL_HOST_DEVICE_FN constexpr bool empty() const { return base_ == nullptr; }

#if defined(__CUDACC__)
  /// Acquire-observe whether this PE has been asked to stop Fabric progress.
  /// \return True after the monotonic shutdown command is published.
  XPOOL_DEVICE_FN bool shutdown_requested() const {
    return xpool::atomic::load_acquire(state().shutdown) != 0;
  }
  /// Publish the monotonic PE-local shutdown command.
  XPOOL_DEVICE_FN void request_shutdown() const {
    xpool::atomic::store_release(state().shutdown, std::uint32_t{1});
  }
  /// Return immutable root geometry at offset zero.
  /// \return Validated Fabric arena layout.
  XPOOL_DEVICE_FN const FabricArenaLayout &layout() const;
  /// Return mutable PE-local arena state.
  /// \return State region owned by the local PE.
  XPOOL_DEVICE_FN FabricArenaState &state() const;
  /// Return one model layout after fail-stop bounds checking.
  /// \param model_index Canonical model table index.
  /// \return Immutable model geometry.
  XPOOL_DEVICE_FN const FabricModelLayout &model_layout(std::size_t model_index) const;
  /// Return one layer layout after fail-stop bounds checking.
  /// \param layer_index Canonical flattened layer table index.
  /// \return Immutable layer identity and kind.
  XPOOL_DEVICE_FN const FabricLayerLayout &layer_layout(std::size_t layer_index) const;
  /// Return the Scheduler object owned by the Coordinator PE.
  /// \return Mutable generation-scoped Scheduler.
  XPOOL_DEVICE_FN FfnScheduler &scheduler() const;
  /// Return one model-scoped Scheduler entry.
  /// \param model_index Canonical model table index.
  /// \return Mutable Scheduler entry for that model.
  XPOOL_DEVICE_FN FfnSchedulerEntry &scheduler_entry(std::size_t model_index) const;
  /// Return one AtnAgent/model Submission publication.
  /// \param atnagent_index Canonical AtnAgent PE index.
  /// \param model_index Canonical model table index.
  /// \return Mutable Submission publication cell.
  XPOOL_DEVICE_FN FabricPublication<FfnSubmission> &submission_publication(std::size_t atnagent_index,
                                                                          std::size_t model_index) const;
  /// Return one AtnAgent/model Admission publication.
  /// \param atnagent_index Canonical AtnAgent PE index.
  /// \param model_index Canonical model table index.
  /// \return Mutable execution-admission publication cell.
  XPOOL_DEVICE_FN FabricPublication<FfnExecutionAdmission> &admission_publication(std::size_t atnagent_index,
                                                                                  std::size_t model_index) const;
  /// Return one Executor Invocation publication.
  /// \param executor_index Distributed Executor index.
  /// \return Mutable Invocation publication cell.
  XPOOL_DEVICE_FN FabricPublication<FfnInvocation> &invocation_publication(std::size_t executor_index) const;
  /// Return one Executor InputReady publication.
  /// \param executor_index Distributed Executor index.
  /// \return Mutable InputReady publication cell.
  XPOOL_DEVICE_FN FabricPublication<FfnInputReady> &input_ready_publication(std::size_t executor_index) const;
  /// Return one FfnAgent/Executor Completion publication.
  /// \param ffnagent_index Canonical FfnAgent index, excluding AtnAgent PEs.
  /// \param executor_index Distributed Executor index.
  /// \return Mutable FfnAgent completion publication cell.
  XPOOL_DEVICE_FN FabricPublication<FfnAgentCompletion> &
  ffnagent_completion_publication(std::size_t ffnagent_index, std::size_t executor_index) const;
  /// Return one AtnAgent/model Result publication.
  /// \param atnagent_index Canonical AtnAgent PE index.
  /// \param model_index Canonical model table index.
  /// \return Mutable result publication cell.
  XPOOL_DEVICE_FN FabricPublication<FfnResult> &result_publication(std::size_t atnagent_index,
                                                                  std::size_t model_index) const;
  /// Return one AtnAgent/model Acknowledgement publication.
  /// \param atnagent_index Canonical AtnAgent PE index.
  /// \param model_index Canonical model table index.
  /// \return Mutable result-acknowledgement publication cell.
  XPOOL_DEVICE_FN FabricPublication<FfnResultAcknowledgement> &
  acknowledgement_publication(std::size_t atnagent_index, std::size_t model_index) const;
  /// Return one model-owned Decode input byte range.
  /// \param model_index Canonical model table index.
  /// \return Start of this model's fixed Decode input payload.
  XPOOL_DEVICE_FN std::uint8_t *model_input_payload(std::size_t model_index) const;
  /// Return one model-owned Decode output byte range.
  /// \param model_index Canonical model table index.
  /// \return Start of this model's fixed Decode output payload.
  XPOOL_DEVICE_FN std::uint8_t *model_output_payload(std::size_t model_index) const;
  /// Return one Executor-owned Prefill input byte range.
  /// \param executor_index Distributed Executor index.
  /// \return Start of this Executor's Prefill input payload.
  XPOOL_DEVICE_FN std::uint8_t *executor_input_payload(std::size_t executor_index) const;
  /// Return one Executor-owned Prefill output byte range.
  /// \param executor_index Distributed Executor index.
  /// \return Start of this Executor's Prefill output payload.
  XPOOL_DEVICE_FN std::uint8_t *executor_output_payload(std::size_t executor_index) const;
  /// Reserve one PE-local Fabric trace entry.
  /// \return Reserved entry, or an empty entry when disabled or full.
  XPOOL_DEVICE_FN xpool::trace::Entry<FabricTraceRecord> reserve_trace() const;
  /// Find one retained PE-local Fabric trace by monotonic identity.
  /// \param local_trace_id Positive local trace sequence.
  /// \return Retained record, or nullptr when unavailable.
  XPOOL_DEVICE_FN FabricTraceRecord *find_trace(std::uint64_t local_trace_id) const;
#endif

private:
#if defined(__CUDACC__)
  template <typename T> XPOOL_DEVICE_FN T *pointer_at(std::size_t offset, std::size_t index = 0) const;
#endif
  std::uint8_t *base_ = nullptr;
};

/// Explicit owner of one process-local NVSHMEM symmetric arena allocation.
class FabricArena {
public:
  /// Construct an empty arena owner.
  FabricArena() = default;
  /// Fail-stop when coordinated shutdown leaves a live symmetric allocation.
  ~FabricArena() { xpool::abort_if(base_ != nullptr); }

  /// Return whether this owner contains a live symmetric allocation.
  explicit operator bool() const noexcept { return base_ != nullptr; }

  FabricArena(const FabricArena &) = delete;
  FabricArena &operator=(const FabricArena &) = delete;

  /// Move one arena owner and leave the source empty.
  /// \param other Arena owner whose resources should be transferred.
  FabricArena(FabricArena &&other) noexcept
      : base_(std::exchange(other.base_, nullptr)), layout_(std::exchange(other.layout_, {})),
        model_topologies_(std::move(other.model_topologies_)) {}

  /// Replace this empty owner by moving another arena owner.
  /// \param other Arena owner whose resources should be transferred.
  /// \return This owner after transfer.
  FabricArena &operator=(FabricArena &&other) {
    if (this != &other) {
      TORCH_CHECK(base_ == nullptr, "a live Fabric arena cannot be replaced by move");
      base_ = std::exchange(other.base_, nullptr);
      layout_ = std::exchange(other.layout_, {});
      model_topologies_ = std::move(other.model_topologies_);
    }
    return *this;
  }

  /// Validate, allocate, initialize, and return one symmetric arena owner.
  /// \param layout Validated root layout shared by every PE.
  /// \param models Canonical model layouts stored in the arena.
  /// \param layers Flattened canonical layer layouts stored in the arena.
  /// \param scheduler_policy Immutable Scheduler policy for the generation.
  /// \return Owner of the process-local symmetric allocation.
  static FabricArena create(const FabricArenaLayout &layout, std::span<const FabricModelLayout> models,
                            std::span<const FabricLayerLayout> layers,
                            const FfnSchedulerPolicy &scheduler_policy);

  /// Return the immutable host-cached root layout.
  /// \return Validated root layout for this allocation.
  const FabricArenaLayout &layout() const {
    TORCH_CHECK(base_ != nullptr, "xpool cannot read layout from an empty Fabric arena");
    return layout_;
  }

  /// Copy this PE's mutable arena state to host memory.
  /// \return Host snapshot of the local mutable state.
  FabricArenaState state() const;

  /// Boundedly wait for the FfnAgent Resident's device startup publication.
  /// \param resident_stream Stream owning the cooperative Resident kernel.
  /// \param timeout Maximum monotonic host duration allowed for startup.
  /// \throws c10::Error on premature Resident completion, timeout, an invalid
  /// readiness publication, or CUDA copy failure.
  void wait_until_resident_ready(
      const xpool::utils::device::OwnedCudaStream &resident_stream,
      std::chrono::steady_clock::duration timeout) const;

  /// Return a typed non-owning address view.
  /// \return View over this owner's process-local arena address.
  FabricArenaView view() const { return FabricArenaView{base_}; }

  /// Publish the monotonic PE-local shutdown command on a control stream.
  /// \param stream Live CUDA stream owner used for the asynchronous device write.
  void request_shutdown(const xpool::utils::device::OwnedCudaStream &stream) const;

  /// Copy this drained PE's retained local Fabric trace rows.
  /// \return Host-owned trace snapshot in local sequence order.
  FabricTraceSnapshot read_trace() const;

  /// Collectively release this arena and reset the owner to empty.
  void destroy();

private:
  /// Construct the owner returned after successful collective initialization.
  FabricArena(std::uint8_t *base, FabricArenaLayout layout,
              std::vector<FabricTraceModelTopology> model_topologies)
      : base_(base), layout_(std::move(layout)), model_topologies_(std::move(model_topologies)) {}

  std::uint8_t *base_ = nullptr;
  FabricArenaLayout layout_{};
  std::vector<FabricTraceModelTopology> model_topologies_;
};

static_assert(std::is_trivially_copyable_v<FabricArenaState>);
static_assert(std::is_trivially_copyable_v<FabricArenaView>);

} // namespace xpool::fabric
