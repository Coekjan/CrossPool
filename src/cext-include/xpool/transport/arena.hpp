#pragma once

/// \file xpool/transport/arena.hpp
/// \brief Transport arena state, address view, and host ownership.

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <cuda/atomic>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include <xpool/ffn.hpp>
#include <xpool/macros.hpp>
#include <xpool/transport/layout.hpp>
#include <xpool/transport/protocol.hpp>
#include <xpool/utils/hex.hpp>

namespace xpool::transport {

/// Opaque binary CUDA IPC handle with canonical hexadecimal projection.
using ArenaHandle = xpool::utils::hex::HexValue<cudaIpcMemHandle_t>;

/// Mutable IPC-visible state owned by one Transport arena.
struct ArenaState {
  /// Monotonic shutdown fact published by the AtnAgent Resident block.
  std::uint32_t shutdown;
  /// Sticky canonical Fabric generation failure copied by the AtnAgent.
  xpool::ffn::ResultCode generation_failure_code;
};

/// Process-local control state shared by every block in one Transport Resident.
///
/// This object lives in AtnAgent-owned device storage and is never exported
/// through an IPC handle or included in an arena layout.
struct ResidentState {
  /// Monotonic host/device command requesting whole-grid terminal drain.
  std::uint32_t drain_requested = 0;

#if defined(__CUDACC__)
  /// Acquire-observe whether whole-grid drain has been requested.
  XPOOL_DEVICE_FN bool draining() const {
    return cuda::atomic_ref{drain_requested}.load(cuda::memory_order_acquire) != 0;
  }
  /// Publish the monotonic whole-grid drain command.
  XPOOL_DEVICE_FN void request_drain() {
    cuda::atomic_ref{drain_requested}.store(std::uint32_t{1}, cuda::memory_order_release);
  }
#endif
};

/// Typed non-owning device view over one mapped Transport arena.
struct ArenaView {
  /// Construct an empty view.
  XPOOL_HOST_DEVICE_FN constexpr ArenaView() = default;
  /// Construct a view over one arena base address.
  XPOOL_HOST_DEVICE_FN explicit constexpr ArenaView(std::uint8_t *base) : base_(base) {}
  /// Return whether this view has no arena address.
  XPOOL_HOST_DEVICE_FN constexpr bool empty() const { return base_ == nullptr; }

#if defined(__CUDACC__)
  /// Acquire-observe whether the Resident published arena shutdown.
  XPOOL_DEVICE_FN bool shutdown_requested() const {
    return cuda::atomic_ref{state().shutdown}.load(cuda::memory_order_acquire) != 0;
  }
  /// Publish this arena's monotonic terminal shutdown fact.
  XPOOL_DEVICE_FN void publish_shutdown() const {
    cuda::atomic_ref{state().shutdown}.store(std::uint32_t{1}, cuda::memory_order_release);
  }
  /// Return the required immutable arena layout.
  /// \pre !empty() and the owner validated the immutable layout.
  XPOOL_DEVICE_FN const ArenaLayout &layout() const;
  /// Return the required mutable arena state.
  XPOOL_DEVICE_FN ArenaState &state() const;
  /// Return the required request mailbox.
  XPOOL_DEVICE_FN Mailbox &mailbox() const;
  /// Return the request input byte range.
  XPOOL_DEVICE_FN cuda::std::span<std::uint8_t> input_payload() const;
  /// Return the request output byte range.
  XPOOL_DEVICE_FN cuda::std::span<std::uint8_t> output_payload() const;
  /// Return optional physical row spans per DP rank.
  /// The span is empty when the topology has one DP rank.
  XPOOL_DEVICE_FN cuda::std::span<std::uint32_t> dp_rank_payload_rows() const;
  /// Acquire-observe the cached canonical generation failure.
  XPOOL_DEVICE_FN xpool::ffn::ResultCode generation_failure() const;
  /// Publish one canonical Fabric failure into the sticky arena cache.
  XPOOL_DEVICE_FN void publish_generation_failure(xpool::ffn::ResultCode result_code) const;
#endif

private:
#if defined(__CUDACC__)
  template <typename T> XPOOL_DEVICE_FN T *pointer_at(std::size_t offset, std::size_t index = 0) const;
#endif
  std::uint8_t *base_ = nullptr;
};

/// Move-only owner of one CUDA Transport allocation or IPC attachment.
/// Owned mappings may export handles; attached mappings may not. Live
/// operations validate this ownership and surface CUDA failures as c10::Error.
class Arena {
public:
  /// Construct an empty arena owner.
  Arena() = default;
  /// Return whether this owner contains a live mapping.
  explicit operator bool() const noexcept { return base_ != nullptr; }
  /// Best-effort fallback cleanup for a live mapping.
  ~Arena() noexcept {
    try {
      destroy();
    } catch (...) {
    }
  }

  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;
  /// Move one arena owner and leave its source empty.
  Arena(Arena &&other) noexcept
      : base_(std::exchange(other.base_, nullptr)), kind_(other.kind_), layout_(std::exchange(other.layout_, {})) {}
  /// Replace this empty owner by moving another arena owner.
  /// \throws c10::Error when the destination already owns a live mapping.
  Arena &operator=(Arena &&other) {
    if (this != &other) {
      TORCH_CHECK(base_ == nullptr, "a live transport arena cannot be replaced by move");
      base_ = std::exchange(other.base_, nullptr);
      kind_ = other.kind_;
      layout_ = std::exchange(other.layout_, {});
    }
    return *this;
  }

  /// Allocate and initialize one AtnAgent-owned CUDA IPC arena.
  static Arena create(c10::DeviceIndex cuda_device, const ArenaLayout &layout);
  /// Open one Instance-side CUDA IPC mapping.
  static Arena from_handle(const ArenaHandle &handle);
  /// Return a typed non-owning view over this mapping.
  ArenaView view() const { return ArenaView{base_}; }
  /// Return the CUDA device that owns this mapping.
  c10::DeviceIndex cuda_device() const;
  /// Release the owned allocation or attached IPC mapping.
  /// The release operation is selected by the mapping's ownership origin.
  void destroy();
  /// Export the CUDA IPC handle for an owned arena.
  /// \throws c10::Error for an empty or attached mapping, or CUDA export failure.
  ArenaHandle handle() const;
  /// Return the immutable host-cached arena layout.
  const ArenaLayout &layout() const {
    TORCH_CHECK(base_ != nullptr, "xpool cannot read layout from an empty transport arena");
    return layout_;
  }
  /// Copy the current arena-local state to host memory.
  ArenaState state() const;
  /// Acquire the current mailbox lifecycle value from host code.
  MailboxStatus mailbox_status() const;
  /// Read the sticky canonical generation failure cache.
  xpool::ffn::ResultCode read_generation_failure() const { return state().generation_failure_code; }

private:
  /// Release behavior selected by the mapping's ownership origin.
  enum class Kind { Owned, Attached };
  /// Construct one validated live arena owner.
  Arena(std::uint8_t *base, Kind kind, ArenaLayout layout) : base_(base), kind_(kind), layout_(std::move(layout)) {}

  std::uint8_t *base_ = nullptr;
  Kind kind_ = Kind::Owned;
  ArenaLayout layout_{};
};

} // namespace xpool::transport
