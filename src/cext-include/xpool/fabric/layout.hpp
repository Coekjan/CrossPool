#pragma once

/// \file xpool/fabric/layout.hpp
/// \brief Immutable geometry for one generation-scoped NVSHMEM Fabric arena.

#include <cstddef>
#include <cstdint>

#include <xpool/abort.hpp>
#include <xpool/arena.hpp>
#include <xpool/macros.hpp>
#include <xpool/trace.hpp>
#include <xpool/utils/enum.hpp>

namespace xpool::fabric {

/// Required alignment of each remotely signaled Fabric publication.
inline constexpr std::size_t kFabricPublicationAlignment = 256;

/// Magic identifying a Fabric arena root at symmetric allocation offset zero.
inline constexpr std::uint64_t kFabricArenaMagic = 0x3141464c4f4f5058ULL;

/// Structural FFN layer kind stored in the canonical Fabric layer table.
class FfnLayerKind {
public:
  /// Stable structural layer-kind values.
  enum Type : std::uint32_t {
    /// Dense feed-forward network implementation.
    Dense = 1,
    /// Sparse mixture-of-experts implementation.
    Sparse = 2,
  };

  /// Construct a validated layer kind from its same-domain enum.
  /// \param value Stable structural layer kind.
  XPOOL_HOST_DEVICE_FN constexpr FfnLayerKind(Type value) : FfnLayerKind(static_cast<std::uint32_t>(value)) {}

  /// Explicitly construct a validated layer kind from an integer.
  /// \tparam T This type's enum or a non-boolean integral input.
  /// \param value Stable structural layer kind.
  /// \pre is_valid(value) is true; violation fail-stops.
  template <xpool::utils::EnumInput<Type> T>
  XPOOL_HOST_DEVICE_FN explicit constexpr FfnLayerKind(T value) : value_(static_cast<Type>(value)) {
    if (!is_valid(value)) {
      xpool::abort();
    }
  }

  /// Return the wrapped structural layer kind.
  /// \return Stable layer-kind enum value.
  XPOOL_HOST_DEVICE_FN constexpr Type value() const { return value_; }

  /// Return whether a raw value names a supported layer kind.
  /// \tparam T Same-domain enum or non-boolean integral input.
  /// \param value Candidate layer-kind representation.
  /// \return True when value denotes Dense or Sparse.
  template <xpool::utils::EnumInput<Type> T> XPOOL_HOST_DEVICE_FN static constexpr bool is_valid(T value) {
    return xpool::utils::enum_value_equal(value, Dense) || xpool::utils::enum_value_equal(value, Sparse);
  }

  /// Compare two validated layer kinds.
  /// \return True when both wrappers contain the same kind.
  constexpr bool operator==(const FfnLayerKind &) const = default;

  /// Compare this wrapper with one same-domain enum value.
  /// \param value Layer kind to compare.
  /// \return True when this wrapper contains value.
  constexpr bool operator==(Type value) const { return value_ == value; }

private:
  Type value_;
};

/// Immutable payload and topology geometry for one configured model.
struct FabricModelLayout {
  /// Raw abi::TensorDType value shared by every model payload.
  std::uint32_t dtype;
  /// Hidden-state columns in every invocation.
  std::size_t hidden_size;
  /// Tensor-parallel AtnAgent width for this model.
  std::size_t atn_tp_size;
  /// Data-parallel AtnAgent width for this model.
  std::size_t atn_dp_size;
  /// First record in the flattened canonical layer table.
  std::size_t layer_begin;
  /// Number of canonical layer records owned by this model.
  std::size_t layer_count;
  /// Byte offset of this model's fixed Decode payload in each direction.
  std::size_t decode_payload_offset;
  /// Aligned bytes available in each model-owned Decode payload.
  std::size_t decode_payload_capacity_bytes;
  /// Aligned Executor payload capacity required by this model's Prefill requests.
  std::size_t prefill_payload_capacity_bytes;

  /// Flatten one TP/DP coordinate using TP-fastest ordering.
  /// \param atn_tp_rank Tensor-parallel coordinate.
  /// \param atn_dp_rank Data-parallel coordinate.
  /// \return Canonical AtnAgent index for the coordinate pair.
  XPOOL_HOST_DEVICE_FN std::size_t atnagent_index(std::size_t atn_tp_rank, std::size_t atn_dp_rank) const {
    xpool::abort_if(atn_tp_rank >= atn_tp_size || atn_dp_rank >= atn_dp_size);
    return atn_dp_rank * atn_tp_size + atn_tp_rank;
  }

  /// Return the TP coordinate for one flattened AtnAgent index.
  /// \param atnagent_index Canonical TP-fastest AtnAgent index.
  /// \return Tensor-parallel coordinate.
  XPOOL_HOST_DEVICE_FN std::size_t atn_tp_rank(std::size_t atnagent_index) const {
    xpool::abort_if(atn_tp_size == 0 || atnagent_index >= atn_tp_size * atn_dp_size);
    return atnagent_index % atn_tp_size;
  }

  /// Return the DP coordinate for one flattened AtnAgent index.
  /// \param atnagent_index Canonical TP-fastest AtnAgent index.
  /// \return Data-parallel coordinate.
  XPOOL_HOST_DEVICE_FN std::size_t atn_dp_rank(std::size_t atnagent_index) const {
    xpool::abort_if(atn_tp_size == 0 || atnagent_index >= atn_tp_size * atn_dp_size);
    return atnagent_index / atn_tp_size;
  }

  /// Validate intrinsic model topology and payload geometry.
  void validate() const;

  /// Compare two immutable model layouts.
  /// \return True when every model-layout field is equal.
  constexpr bool operator==(const FabricModelLayout &) const = default;
};

/// One canonical decoder-layer identity in the flattened Fabric layer table.
struct FabricLayerLayout {
  /// Model decoder layer identifier used for executor binding and diagnostics.
  std::size_t layer_id;
  /// Raw FfnLayerKind value selecting the executor family.
  std::uint32_t kind;

  /// Validate intrinsic layer identity and structural kind.
  void validate() const;

  /// Compare two immutable layer layouts.
  /// \return True when layer identity and kind are equal.
  constexpr bool operator==(const FabricLayerLayout &) const = default;
};

/// Immutable root geometry stored at symmetric-arena offset zero.
struct FabricArenaLayout {
  /// Common allocation-root header.
  xpool::arena::LayoutHeader header;
  /// Number of AtnAgent PEs in the canonical prefix.
  std::size_t atnagent_count;
  /// Number of FfnAgent PEs in the canonical suffix.
  std::size_t ffnagent_count;
  /// Number of model records.
  std::size_t model_count;
  /// Number of flattened canonical layer records.
  std::size_t layer_count;
  /// Number of independent distributed FFN Executors.
  std::size_t executor_count;
  /// Byte offset of the FabricModelLayout table.
  std::size_t model_layouts_offset;
  /// Byte offset of the FabricLayerLayout table.
  std::size_t layer_layouts_offset;
  /// Byte offset of the FfnScheduler object.
  std::size_t scheduler_offset;
  /// Byte offset of the model-indexed FfnSchedulerEntry table.
  std::size_t scheduler_entries_offset;
  /// Byte offset of AtnAgent/model Submission publications.
  std::size_t submission_publications_offset;
  /// Byte offset of AtnAgent/model Admission publications.
  std::size_t admission_publications_offset;
  /// Byte offset of Executor Invocation publications.
  std::size_t invocation_publications_offset;
  /// Byte offset of Executor InputReady publications.
  std::size_t input_ready_publications_offset;
  /// Byte offset of FfnAgent/Executor Completion publications.
  std::size_t ffnagent_completion_publications_offset;
  /// Byte offset of AtnAgent/model Result publications.
  std::size_t result_publications_offset;
  /// Byte offset of AtnAgent/model Acknowledgement publications.
  std::size_t acknowledgement_publications_offset;
  /// Byte offset of the first model-owned Decode input payload.
  std::size_t model_input_payloads_offset;
  /// Byte offset of the first model-owned Decode output payload.
  std::size_t model_output_payloads_offset;
  /// Byte offset of the first Executor-owned Prefill input payload.
  std::size_t executor_input_payloads_offset;
  /// Byte offset of the first Executor-owned Prefill output payload.
  std::size_t executor_output_payloads_offset;
  /// Aligned payload capacity owned by each distributed Executor.
  std::size_t executor_payload_capacity_bytes;
  /// Geometry of this PE's local Fabric trace buffer.
  xpool::trace::BufferLayout trace;

  /// Return the first FfnAgent PE and sole Coordinator owner.
  /// \return NVSHMEM PE index of the generation Coordinator.
  XPOOL_HOST_DEVICE_FN int coordinator_pe() const { return static_cast<int>(atnagent_count); }

  /// Build root geometry from validated topology and aggregate capacities.
  /// \param atnagent_count Number of AtnAgent PEs in the canonical prefix.
  /// \param ffnagent_count Number of FfnAgent PEs in the canonical suffix.
  /// \param executor_count Number of distributed FFN Executors.
  /// \param model_count Number of canonical model records.
  /// \param layer_count Number of flattened canonical layer records.
  /// \param model_payloads_bytes Aggregate bytes for all fixed Decode payloads.
  /// \param executor_payload_capacity_bytes Per-Executor Prefill payload capacity.
  /// \return Validated aligned root layout with debug-derived trace geometry.
  static FabricArenaLayout create(std::size_t atnagent_count, std::size_t ffnagent_count,
                                  std::size_t executor_count, std::size_t model_count, std::size_t layer_count,
                                  std::size_t model_payloads_bytes,
                                  std::size_t executor_payload_capacity_bytes);

  /// Validate this arena's immutable ABI and region geometry.
  void validate() const;

  /// Compare two immutable Fabric arena layouts.
  /// \return True when every root-layout field is equal.
  constexpr bool operator==(const FabricArenaLayout &) const = default;
};

} // namespace xpool::fabric
