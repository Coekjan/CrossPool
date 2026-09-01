#pragma once

/// \file xpool/fabric/arena.hpp
/// \brief Fabric symmetric-arena state, typed views, and explicit owner.

#include <cuda/atomic>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <array>
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
#include <xpool/macros.hpp>
#include <xpool/utils/device.hpp>

namespace xpool::fabric {

/// Mutable device state local to one PE's symmetric arena allocation.
struct ArenaState {
  /// Nonzero after this PE has requested resident-kernel shutdown.
  std::uint32_t shutdown;
  /// First Fabric failure published for this PE.
  Failure failure;
};

/// Non-owning interpretation of one lane's fixed-capacity routing block.
class RoutingMetadataBlock {
public:
#if defined(__CUDACC__)
  /// Return writable storage for the live top-k expert identifiers.
  XPOOL_DEVICE_FN cuda::std::span<std::int32_t> topk_ids_destination() const;
  /// Return the live top-k expert identifiers.
  XPOOL_DEVICE_FN cuda::std::span<const std::int32_t> topk_ids() const;
  /// Return writable storage for the live top-k weights.
  XPOOL_DEVICE_FN cuda::std::span<float> topk_weights_destination() const;
  /// Return the live top-k weights.
  XPOOL_DEVICE_FN cuda::std::span<const float> topk_weights() const;
  /// Return the entire fixed-capacity block used by remote publication.
  XPOOL_DEVICE_FN cuda::std::span<std::uint8_t> publication_payload() const;
#endif

private:
  friend struct ArenaView;
  XPOOL_HOST_DEVICE_FN constexpr RoutingMetadataBlock(std::uint8_t *base, std::size_t capacity_elements,
                                                      std::size_t live_elements)
      : base_(base), capacity_elements_(capacity_elements), live_elements_(live_elements) {}

  std::uint8_t *base_;
  std::size_t capacity_elements_;
  std::size_t live_elements_;
};

/// AtnAgent interpretation of one lane's two physical payload regions.
class AtnAgentLanePayloadView {
public:
#if defined(__CUDACC__)
  /// Return writable storage for the input payload produced by this AtnAgent.
  XPOOL_DEVICE_FN cuda::std::span<std::uint8_t> input_destination() const;
  /// Return the input payload produced by this AtnAgent.
  XPOOL_DEVICE_FN cuda::std::span<const std::uint8_t> input() const;
  /// Return writable storage for the completed output payload.
  XPOOL_DEVICE_FN cuda::std::span<std::uint8_t> output_destination() const;
  /// Return the completed output payload consumed by this AtnAgent.
  XPOOL_DEVICE_FN cuda::std::span<const std::uint8_t> output() const;
#endif

private:
  friend struct ArenaView;
  XPOOL_HOST_DEVICE_FN constexpr AtnAgentLanePayloadView(
      std::array<std::uint8_t *, kExecutorLanePayloadBufferCount> buffers, std::size_t capacity)
      : buffers_(buffers), capacity_(capacity) {}

  std::array<std::uint8_t *, kExecutorLanePayloadBufferCount> buffers_;
  std::size_t capacity_;
};

/// FfnAgent interpretation of one lane's two physical payload regions.
class FfnAgentLanePayloadView {
public:
#if defined(__CUDACC__)
  /// Return the input payload consumed by this FfnAgent.
  XPOOL_DEVICE_FN cuda::std::span<const std::uint8_t> input() const;
  /// Return writable storage for this FfnAgent's partial result.
  XPOOL_DEVICE_FN cuda::std::span<std::uint8_t> partial_destination() const;
  /// Return this FfnAgent's partial result.
  XPOOL_DEVICE_FN cuda::std::span<const std::uint8_t> partial() const;
  /// Return writable staging storage for a complete output before publication.
  XPOOL_DEVICE_FN cuda::std::span<std::uint8_t> complete_output_staging_destination() const;
  /// Return the staged complete output.
  XPOOL_DEVICE_FN cuda::std::span<const std::uint8_t> complete_output_staging() const;
#endif

private:
  friend struct ArenaView;
  XPOOL_HOST_DEVICE_FN constexpr FfnAgentLanePayloadView(
      std::array<std::uint8_t *, kExecutorLanePayloadBufferCount> buffers, std::size_t capacity)
      : buffers_(buffers), capacity_(capacity) {}

  std::array<std::uint8_t *, kExecutorLanePayloadBufferCount> buffers_;
  std::size_t capacity_;
};

/// Non-owning typed address view over one process-local symmetric arena.
/// Device accessors require a nonempty view whose owner validated and
/// materialized the immutable layout.
struct ArenaView {
  /// Construct an empty view.
  XPOOL_HOST_DEVICE_FN constexpr ArenaView() = default;
  /// Construct a view over a symmetric arena allocation.
  XPOOL_HOST_DEVICE_FN explicit constexpr ArenaView(std::uint8_t *base) : base_(base) {}
  /// Return whether this view is detached from an allocation.
  XPOOL_HOST_DEVICE_FN constexpr bool empty() const { return base_ == nullptr; }

  /// Return host-visible storage for one lane's FFN input payload.
  XPOOL_HOST_FN std::uint8_t *ffn_input_storage(const ArenaLayout &layout, std::size_t executor_lane_index) const {
    return base_ + layout.lane_payload_storage_offset_bytes + executor_lane_index * layout.lane_payload_capacity_bytes;
  }

  /// Return host-visible storage for one lane's FFN partial payload.
  XPOOL_HOST_FN std::uint8_t *ffn_partial_storage(const ArenaLayout &layout, std::size_t executor_lane_index) const {
    return base_ + layout.lane_payload_storage_offset_bytes +
           (layout.executor_lane_count + executor_lane_index) * layout.lane_payload_capacity_bytes;
  }

  /// Return host-visible storage for one lane's routing metadata.
  XPOOL_HOST_FN std::uint8_t *routing_metadata_storage(const ArenaLayout &layout,
                                                       std::size_t executor_lane_index) const {
    return base_ + layout.routing_metadata_offset_bytes + executor_lane_index * layout.routing_metadata_stride_bytes;
  }

  /// Return the host-visible payload-row field captured by a lane graph.
  XPOOL_HOST_FN std::size_t *lane_payload_rows(const ArenaLayout &layout, std::size_t executor_lane_index) const {
    const auto publication_bytes = sizeof(Publication<LaneExecution>);
    const auto record_offset = offsetof(Publication<LaneExecution>, record);
    return reinterpret_cast<std::size_t *>(base_ + layout.lane_execution_publications_offset_bytes +
                                           executor_lane_index * publication_bytes + record_offset +
                                           offsetof(LaneExecution, payload_rows));
  }

#if defined(__CUDACC__)
  /// Return whether shutdown has been published for this PE.
  /// The result is acquire-loaded from the device-local arena state.
  XPOOL_DEVICE_FN bool shutdown_requested() const {
    return cuda::atomic_ref{state().shutdown}.load(cuda::memory_order_acquire) != 0;
  }
  /// Publish a shutdown request for this PE.
  XPOOL_DEVICE_FN void request_shutdown() const {
    cuda::atomic_ref{state().shutdown}.store(std::uint32_t{1}, cuda::memory_order_release);
  }
  /// Return the protocol result that should terminate an in-flight request.
  /// Returns the first Fabric failure, or cancellation when none was recorded.
  XPOOL_DEVICE_FN xpool::ffn::ResultCode cancellation_result() const;

  /// Return the immutable layout copied into the allocation header.
  XPOOL_DEVICE_FN const ArenaLayout &layout() const;
  /// Return this PE's mutable arena state.
  XPOOL_DEVICE_FN ArenaState &state() const;
  /// Return one materialized model-instance entry.
  XPOOL_DEVICE_FN const InstanceEntry &instance_entry(std::size_t instance_index) const;
  /// Return one materialized model-layer entry.
  XPOOL_DEVICE_FN const LayerEntry &layer_entry(std::size_t instance_index, std::size_t layer_ordinal) const;
  /// Return the AtnAgent PE membership for one instance.
  XPOOL_DEVICE_FN cuda::std::span<const int> atnagent_pes(std::size_t instance_index) const;
  /// Return the FfnAgent PE membership for one layer.
  XPOOL_DEVICE_FN cuda::std::span<const int> ffnagent_pes(std::size_t instance_index, std::size_t layer_ordinal) const;
  /// Interpret one lane's routing metadata for the live row capacity.
  XPOOL_DEVICE_FN RoutingMetadataBlock routing_metadata(std::size_t executor_lane_index,
                                                        std::size_t payload_row_capacity) const;

  /// Return the submission publication owned by one source AtnAgent.
  XPOOL_DEVICE_FN Publication<Submission> &submission_publication(std::size_t source_atnagent_index,
                                                                  std::size_t instance_index) const;
  /// Return the admission publication for one instance.
  XPOOL_DEVICE_FN Publication<Admission> &admission_publication(std::size_t instance_index) const;
  /// Return the execution publication for one lane.
  XPOOL_DEVICE_FN Publication<LaneExecution> &lane_execution_publication(std::size_t executor_lane_index) const;
  /// Return the input-readiness publication for one lane.
  XPOOL_DEVICE_FN Publication<InputReady> &input_ready_publication(std::size_t executor_lane_index) const;
  /// Return the routing-metadata readiness publication for one lane.
  XPOOL_DEVICE_FN Publication<RoutingMetadataReady> &
  routing_metadata_ready_publication(std::size_t executor_lane_index) const;
  /// Return one FfnAgent's partial-readiness publication for a lane.
  XPOOL_DEVICE_FN Publication<PartialReady> &partial_ready_publication(std::size_t source_ffnagent_index,
                                                                       std::size_t executor_lane_index) const;
  /// Return one FfnAgent's completion publication for a lane.
  XPOOL_DEVICE_FN Publication<FfnAgentCompletion> &
  ffnagent_completion_publication(std::size_t source_ffnagent_index, std::size_t executor_lane_index) const;
  /// Return the output-commit publication for one instance.
  XPOOL_DEVICE_FN Publication<OutputCommit> &output_commit_publication(std::size_t instance_index) const;
  /// Return one AtnAgent's output acknowledgement publication.
  XPOOL_DEVICE_FN Publication<OutputAcknowledgement> &
  output_acknowledgement_publication(std::size_t source_atnagent_index, std::size_t instance_index) const;

  /// Interpret one lane's physical payload buffers from the AtnAgent role.
  XPOOL_DEVICE_FN AtnAgentLanePayloadView atnagent_lane_payload(std::size_t executor_lane_index) const;
  /// Interpret one lane's physical payload buffers from the FfnAgent role.
  XPOOL_DEVICE_FN FfnAgentLanePayloadView ffnagent_lane_payload(std::size_t executor_lane_index) const;

#endif

private:
#if defined(__CUDACC__)
  template <typename T> XPOOL_DEVICE_FN T *pointer_at(std::size_t offset, std::size_t index = 0) const;
#endif
  std::uint8_t *base_ = nullptr;
};

#if defined(__CUDACC__)
/// Select the delivery protocol implied by instance topology and output count.
/// \return Complete delivery when every consumer can receive a complete value;
/// otherwise partial delivery.
XPOOL_DEVICE_FN DeliveryVariant delivery_variant(const InstanceEntry &instance,
                                                 xpool::ffn::OutputRequirement output_requirement);
#endif

/// Explicit owner of one process-local NVSHMEM symmetric arena allocation.
/// Live operations require the allocation installed by Fabric Runtime and
/// surface CUDA or NVSHMEM failures as c10::Error.
class Arena {
public:
  /// Construct an empty owner.
  Arena() = default;
  /// Require callers to destroy a live symmetric allocation explicitly.
  ~Arena() { xpool::abort_if(base_ != nullptr); }

  /// Return whether this owner currently holds an allocation.
  explicit operator bool() const noexcept { return base_ != nullptr; }

  /// Symmetric arena ownership cannot be copied.
  Arena(const Arena &) = delete;
  /// Symmetric arena ownership cannot be copied.
  Arena &operator=(const Arena &) = delete;

  /// Transfer ownership from another arena.
  Arena(Arena &&other) noexcept
      : base_(std::exchange(other.base_, nullptr)), layout_(std::exchange(other.layout_, {})) {}

  /// Transfer ownership into an empty arena.
  /// \throws c10::Error when the destination already owns a live allocation.
  Arena &operator=(Arena &&other) {
    if (this != &other) {
      TORCH_CHECK(base_ == nullptr, "a live Fabric arena cannot be replaced by move");
      base_ = std::exchange(other.base_, nullptr);
      layout_ = std::exchange(other.layout_, {});
    }
    return *this;
  }

  /// Return the materialized layout of a live arena.
  const ArenaLayout &layout() const {
    TORCH_CHECK(base_ != nullptr, "xpool cannot read layout from an empty Fabric arena");
    return layout_;
  }

  /// Copy this PE's current device state to the host.
  ArenaState state() const;
  /// Return a non-owning address view over this allocation.
  ArenaView view() const { return ArenaView{base_}; }
  /// Publish shutdown on the supplied stream.
  void request_shutdown(const xpool::utils::device::OwnedCudaStream &stream) const;
  /// Release the symmetric allocation and reset this owner.
  /// \pre No resident or graph execution still addresses the allocation.
  void destroy();

private:
  friend class Runtime;

  /// Allocate an arena from tables materialized by a validated Runtime join.
  /// \pre layout and every span were produced together from the joined Projection.
  static Arena create(const ArenaLayout &layout, std::span<const InstanceEntry> instances,
                      std::span<const LayerEntry> layers, std::span<const int> atnagent_pes,
                      std::span<const int> ffnagent_pes);

  Arena(std::uint8_t *base, ArenaLayout layout) : base_(base), layout_(std::move(layout)) {}

  std::uint8_t *base_ = nullptr;
  ArenaLayout layout_{};
};

static_assert(std::is_trivially_copyable_v<ArenaState>);
static_assert(std::is_trivially_copyable_v<ArenaView>);

} // namespace xpool::fabric
