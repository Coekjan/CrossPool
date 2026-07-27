#pragma once

/// \file xpool/debug/loopback.cuh
/// \brief Internal CUDA helpers for the debug FFN loopback executor.

#include <ATen/ATen.h>

#include <cstddef>

#include <xpool/abi.hpp>
#include <xpool/abort.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/cooperative.cuh>

namespace xpool::debug {

/// Multiplicative factor for the debug 45-degree hidden-pair rotation.
inline constexpr float kHiddenPairRotationInvSqrt2 = 0.70710678118654752440F;

/// Rotate one hidden-state pair.
/// \tparam scalar_t CUDA scalar type stored by the hidden-state buffer.
/// \param output Device output buffer containing the selected pair.
/// \param input Device input buffer containing the selected pair.
/// \param pair_index Hidden pair to rotate.
template <typename scalar_t>
XPOOL_DEVICE_FN void rotate_hidden_pair(scalar_t *output, const scalar_t *input,
                                                        std::size_t pair_index) {
  const auto base = pair_index * 2;
  const auto x = static_cast<float>(input[base]);
  const auto y = static_cast<float>(input[base + 1]);
  output[base] = static_cast<scalar_t>((x - y) * kHiddenPairRotationInvSqrt2);
  output[base + 1] = static_cast<scalar_t>((x + y) * kHiddenPairRotationInvSqrt2);
}

/// Rotate hidden-state pairs cooperatively.
/// \tparam Group Cooperative group exposing thread_rank() and size().
/// \tparam scalar_t CUDA scalar type stored by the hidden-state buffer.
/// \param group Threads participating in the rotation.
/// \param output Device output buffer with at least pair_count * 2 elements.
/// \param input Device input buffer with at least pair_count * 2 elements.
/// \param pair_count Total number of hidden pairs in the buffer.
template <xpool::utils::cooperative::CooperativeGroup Group, typename scalar_t>
XPOOL_DEVICE_FN void rotate_hidden_pairs(Group group, scalar_t *output, const scalar_t *input,
                                                         std::size_t pair_count) {
  for (auto pair_index = static_cast<std::size_t>(group.thread_rank()); pair_index < pair_count;
       pair_index += static_cast<std::size_t>(group.size())) {
    rotate_hidden_pair(output, input, pair_index);
  }
}

/// Rotate hidden-state pairs stored in a byte-oriented payload.
/// \tparam Group Cooperative group exposing thread_rank() and size().
/// \param group Threads participating in the rotation.
/// \param output Device output buffer with at least pair_count * 2 elements.
/// \param input Device input buffer with at least pair_count * 2 elements.
/// \param dtype Valid hidden-state scalar type shared by both buffers.
/// \param pair_count Total number of hidden pairs in the buffers.
/// \pre dtype is FP32, FP16, or BF16 and both buffers have matching alignment.
template <xpool::utils::cooperative::CooperativeGroup Group>
XPOOL_DEVICE_FN void rotate_hidden_pairs(Group group, void *output, const void *input,
                                                         xpool::abi::TensorDType dtype, std::size_t pair_count) {
  switch (dtype.value()) {
  case xpool::abi::TensorDType::Fp32:
    rotate_hidden_pairs(group, static_cast<float *>(output), static_cast<const float *>(input), pair_count);
    return;
  case xpool::abi::TensorDType::Fp16:
    rotate_hidden_pairs(group, static_cast<at::Half *>(output), static_cast<const at::Half *>(input), pair_count);
    return;
  case xpool::abi::TensorDType::Bf16:
    rotate_hidden_pairs(group, static_cast<at::BFloat16 *>(output), static_cast<const at::BFloat16 *>(input),
                        pair_count);
    return;
  }
  xpool::abort();
}

} // namespace xpool::debug
