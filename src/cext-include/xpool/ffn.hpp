#pragma once

/// \file xpool/ffn.hpp
/// \brief Closed FFN semantics shared by native CrossPool domains.

#include <cstdint>

#include <c10/core/ScalarType.h>

#include <xpool/macros.hpp>

namespace xpool::ffn {

/// Runtime forward mode carried by one FFN request.
enum class ForwardMode : std::uint32_t {
  /// Variable-row prompt ingestion using the shared data-plane protocol.
  Prefill = 1,
  /// Token-step execution using the shared data-plane protocol.
  Decode = 2,
  /// Idle contribution from a DP rank with no rows in this invocation.
  Idle = 4,
};

/// Mathematical output requirement for one FFN invocation.
enum class OutputRequirement : std::uint32_t {
  /// Preserve the rank-local Partial for reduction by the consumer.
  PerRankComplete = 1,
  /// Deliver the sum of every participating FfnAgent Partial.
  GroupSumComplete = 2,
};

/// Physical data-parallel row layout carried by one FFN request.
enum class DpRowLayout : std::uint32_t {
  /// One DP rank owns the complete payload and needs no per-rank row vector.
  None = 0,
  /// Every DP rank occupies an equal-capacity physical span.
  UniformByRank = 1,
  /// DP-rank spans are packed consecutively by their live row counts.
  PackedByRank = 2,
};

/// Result code shared by Transport and Fabric completion paths.
enum class ResultCode : std::uint32_t {
  Ok = 0,
  Shutdown = 1,
  ProtocolMismatch = 2,
  Timeout = 3,
};

/// Structural kind of one gated FFN layer.
enum class LayerKind : std::uint32_t {
  Dense = 1,
  Moe = 2,
};

/// Return whether a forward mode belongs to the closed FFN vocabulary.
XPOOL_HOST_DEVICE_FN constexpr bool is_valid(ForwardMode value) {
  return value == ForwardMode::Prefill || value == ForwardMode::Decode || value == ForwardMode::Idle;
}

/// Return whether an output requirement belongs to the closed FFN vocabulary.
XPOOL_HOST_DEVICE_FN constexpr bool is_valid(OutputRequirement value) {
  return value == OutputRequirement::PerRankComplete || value == OutputRequirement::GroupSumComplete;
}

/// Return whether a DP row layout belongs to the closed FFN vocabulary.
XPOOL_HOST_DEVICE_FN constexpr bool is_valid(DpRowLayout value) {
  return value == DpRowLayout::None || value == DpRowLayout::UniformByRank || value == DpRowLayout::PackedByRank;
}

/// Return whether a result code belongs to the closed FFN vocabulary.
XPOOL_HOST_DEVICE_FN constexpr bool is_valid(ResultCode value) {
  return value == ResultCode::Ok || value == ResultCode::Shutdown || value == ResultCode::ProtocolMismatch ||
         value == ResultCode::Timeout;
}

/// Return whether a layer kind belongs to the closed FFN vocabulary.
XPOOL_HOST_DEVICE_FN constexpr bool is_valid(LayerKind value) {
  return value == LayerKind::Dense || value == LayerKind::Moe;
}

/// Return whether a Torch dtype is admitted for FFN payload execution.
XPOOL_HOST_DEVICE_FN constexpr bool is_supported_payload_dtype(c10::ScalarType value) {
  return value == c10::ScalarType::BFloat16 || value == c10::ScalarType::Half;
}

} // namespace xpool::ffn
