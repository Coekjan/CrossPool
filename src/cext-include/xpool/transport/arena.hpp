#pragma once

/// \file xpool/transport/arena.hpp
/// \brief Transport arena state, address view, and host ownership.

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include <xpool/abi.hpp>
#include <xpool/macros.hpp>
#include <xpool/trace.hpp>
#include <xpool/transport/layout.hpp>
#include <xpool/transport/protocol.hpp>
#include <xpool/transport/trace.hpp>
#include <xpool/utils/hex.hpp>

#if defined(__CUDACC__)
#include <xpool/atomic.cuh>
#endif

namespace xpool::transport {

/// Opaque binary CUDA IPC handle with canonical hexadecimal projection.
using TransportArenaHandle = xpool::utils::hex::HexValue<cudaIpcMemHandle_t>;

/// Mutable IPC-visible state owned by one Transport arena.
struct TransportArenaState {
  /// Monotonic shutdown fact published by the AtnAgent Resident block.
  std::uint32_t shutdown;
  /// Sticky canonical Fabric generation failure copied by the AtnAgent.
  std::uint32_t generation_failure_code;
  /// Monotonic reservation and drop counters for optional Transport tracing.
  xpool::trace::BufferState trace;
};

/// Process-local control state shared by every block in one Transport Resident.
///
/// This object lives in AtnAgent-owned device storage and is never exported
/// through an IPC handle or included in an arena layout.
struct TransportResidentState {
  /// Monotonic host/device command requesting whole-grid terminal drain.
  std::uint32_t drain_requested = 0;

#if defined(__CUDACC__)
  /// Acquire-observe whether whole-grid drain has been requested.
  /// \return True after either the host or a Resident block requests drain.
  XPOOL_DEVICE_FN bool draining() const {
    return xpool::atomic::load_acquire(drain_requested) != 0;
  }
  /// Publish the monotonic whole-grid drain command.
  XPOOL_DEVICE_FN void request_drain() {
    xpool::atomic::store_release(drain_requested, std::uint32_t{1});
  }
#endif
};

/// Typed non-owning device view over one mapped Transport arena.
struct TransportArenaView {
  /// Construct an empty view.
  XPOOL_HOST_DEVICE_FN constexpr TransportArenaView() = default;
  /// Construct a view over one arena base address.
  /// \param base CUDA-accessible arena allocation base.
  XPOOL_HOST_DEVICE_FN explicit constexpr TransportArenaView(std::uint8_t *base) : base_(base) {}
  /// Return whether this view has no arena address.
  /// \return True for an empty view.
  XPOOL_HOST_DEVICE_FN constexpr bool empty() const { return base_ == nullptr; }

#if defined(__CUDACC__)
  /// Acquire-observe whether the Resident published arena shutdown.
  /// \return True after this arena enters terminal shutdown.
  XPOOL_DEVICE_FN bool shutdown_requested() const {
    return xpool::atomic::load_acquire(state().shutdown) != 0;
  }
  /// Publish this arena's monotonic terminal shutdown fact.
  XPOOL_DEVICE_FN void publish_shutdown() const {
    xpool::atomic::store_release(state().shutdown, std::uint32_t{1});
  }
  /// Return the required immutable arena layout.
  /// \return Arena layout at offset zero.
  XPOOL_DEVICE_FN const TransportArenaLayout &layout() const;
  /// Return the required mutable arena state.
  /// \return Arena-local state region.
  XPOOL_DEVICE_FN TransportArenaState &state() const;
  /// Return the required request mailbox.
  /// \return Sole arena mailbox.
  XPOOL_DEVICE_FN TransportMailbox &mailbox() const;
  /// Return the request input byte range.
  /// \return Start of the fixed input payload region.
  XPOOL_DEVICE_FN std::uint8_t *input_payload() const;
  /// Return the request output byte range.
  /// \return Start of the fixed output payload region.
  XPOOL_DEVICE_FN std::uint8_t *output_payload() const;
  /// Return optional per-DP-rank token counts.
  /// \return Count vector, or nullptr for DP one.
  XPOOL_DEVICE_FN std::uint32_t *dp_token_counts() const;
  /// Reserve one optional Transport trace record.
  /// \return Reserved entry, or an empty entry when tracing is disabled/full.
  XPOOL_DEVICE_FN xpool::trace::Entry<TransportTraceRecord> reserve_trace() const;
  /// Find the trace record for the current request sequence.
  /// \return Retained record, or nullptr when unavailable.
  XPOOL_DEVICE_FN TransportTraceRecord *current_trace() const;
  /// Acquire-observe the cached canonical generation failure.
  /// \return Ok or the retained canonical Fabric failure code.
  XPOOL_DEVICE_FN xpool::abi::FfnResultCode generation_failure() const;
  /// Publish one canonical Fabric failure into the sticky arena cache.
  /// \param result_code Canonical convergent failure code.
  XPOOL_DEVICE_FN void publish_generation_failure(xpool::abi::FfnResultCode result_code) const;
#endif

private:
#if defined(__CUDACC__)
  template <typename T> XPOOL_DEVICE_FN T *pointer_at(std::size_t offset, std::size_t index = 0) const;
#endif
  std::uint8_t *base_ = nullptr;
};

/// Move-only owner of one CUDA Transport allocation or IPC attachment.
class TransportArena {
public:
  /// Construct an empty arena owner.
  TransportArena() = default;
  /// Return whether this owner contains a live mapping.
  explicit operator bool() const noexcept { return base_ != nullptr; }
  /// Best-effort fallback cleanup for a live mapping.
  ~TransportArena() noexcept {
    try {
      destroy();
    } catch (...) {
    }
  }

  TransportArena(const TransportArena &) = delete;
  TransportArena &operator=(const TransportArena &) = delete;
  /// Move one arena owner and leave its source empty.
  /// \param other Arena owner to transfer.
  TransportArena(TransportArena &&other) noexcept
      : base_(std::exchange(other.base_, nullptr)), kind_(other.kind_),
        layout_(std::exchange(other.layout_, {})) {}
  /// Replace this empty owner by moving another arena owner.
  /// \param other Arena owner to transfer.
  /// \return This owner.
  TransportArena &operator=(TransportArena &&other) {
    if (this != &other) {
      TORCH_CHECK(base_ == nullptr, "a live transport arena cannot be replaced by move");
      base_ = std::exchange(other.base_, nullptr);
      kind_ = other.kind_;
      layout_ = std::exchange(other.layout_, {});
    }
    return *this;
  }

  /// Allocate and initialize one AtnAgent-owned CUDA IPC arena.
  /// \param cuda_device CUDA device that owns the allocation.
  /// \param layout Validated immutable arena geometry.
  /// \return Owning Transport arena.
  static TransportArena create(c10::DeviceIndex cuda_device, const TransportArenaLayout &layout);
  /// Open one Instance-side CUDA IPC mapping.
  /// \param handle Binary CUDA IPC handle exported by the AtnAgent.
  /// \return Attached Transport arena.
  static TransportArena from_handle(const TransportArenaHandle &handle);
  /// Return a typed non-owning view over this mapping.
  /// \return Empty or live arena view matching this owner.
  TransportArenaView view() const { return TransportArenaView{base_}; }
  /// Return the CUDA device that owns this mapping.
  /// \return CUDA device index.
  c10::DeviceIndex cuda_device() const;
  /// Release the owned allocation or attached IPC mapping.
  void destroy();
  /// Export the CUDA IPC handle for an owned arena.
  /// \return Binary CUDA IPC handle.
  TransportArenaHandle handle() const;
  /// Return the immutable host-cached arena layout.
  /// \return Validated arena layout.
  const TransportArenaLayout &layout() const {
    TORCH_CHECK(base_ != nullptr, "xpool cannot read layout from an empty transport arena");
    return layout_;
  }
  /// Copy the current arena-local state to host memory.
  /// \return Host snapshot of mutable arena state.
  TransportArenaState state() const;
  /// Acquire the current mailbox lifecycle value from host code.
  /// \return Current validated MailboxStatus.
  MailboxStatus mailbox_status() const;
  /// Copy retained Transport trace records to host memory.
  /// \return Host-owned trace snapshot.
  TransportTraceSnapshot read_trace() const;
  /// Read the sticky canonical generation failure cache.
  /// \return Typed canonical generation result.
  xpool::abi::FfnResultCode read_generation_failure() const {
    TORCH_CHECK(base_ != nullptr, "xpool cannot read failure state from an empty transport arena");
    return xpool::abi::FfnResultCode{state().generation_failure_code};
  }

private:
  /// Release behavior selected by the mapping's ownership origin.
  enum class Kind { Owned, Attached };
  /// Construct one validated live arena owner.
  /// \param base CUDA allocation or IPC mapping base.
  /// \param kind Release operation required by the mapping.
  /// \param layout Validated immutable arena geometry.
  TransportArena(std::uint8_t *base, Kind kind, TransportArenaLayout layout)
      : base_(base), kind_(kind), layout_(std::move(layout)) {}

  std::uint8_t *base_ = nullptr;
  Kind kind_ = Kind::Owned;
  TransportArenaLayout layout_{};
};

} // namespace xpool::transport
