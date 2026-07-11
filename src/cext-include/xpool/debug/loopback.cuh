#pragma once

/// \file xpool/debug/loopback.cuh
/// \brief Internal CUDA helpers for the debug FFN loopback executor.

#include <cstdint>

namespace xpool::debug {

/// Multiplicative factor for the debug 45-degree hidden-pair rotation.
inline constexpr float kHiddenPairRotationInvSqrt2 = 0.70710678118654752440F;

/// Rotate hidden-state pairs using a strided device loop.
/// \tparam scalar_t CUDA scalar type stored by the hidden-state buffer.
/// \param output Device output buffer with at least pair_count * 2 elements.
/// \param input Device input buffer with at least pair_count * 2 elements.
/// \param start_pair First hidden pair processed by this CUDA worker.
/// \param pair_stride Distance between pair indices processed by this worker.
/// \param pair_count Total number of hidden pairs in the buffer.
template <typename scalar_t>
__device__ __forceinline__ void
rotate_hidden_pairs_strided(scalar_t *output, const scalar_t *input,
                            std::int64_t start_pair, std::int64_t pair_stride,
                            std::int64_t pair_count) {
  for (std::int64_t pair_index = start_pair; pair_index < pair_count;
       pair_index += pair_stride) {
    std::int64_t base = pair_index * 2;
    float x = static_cast<float>(input[base]);
    float y = static_cast<float>(input[base + 1]);
    output[base] = static_cast<scalar_t>((x - y) * kHiddenPairRotationInvSqrt2);
    output[base + 1] =
        static_cast<scalar_t>((x + y) * kHiddenPairRotationInvSqrt2);
  }
}

} // namespace xpool::debug
