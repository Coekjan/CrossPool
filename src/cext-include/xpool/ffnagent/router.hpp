#pragma once

/// \file xpool/ffnagent/router.hpp
/// \brief Biased BF16 Router computation with caller-owned cuBLASLt storage.

#include <cstddef>

#include <ATen/core/TensorBody.h>

namespace xpool::ffnagent {

/// Fixed cuBLASLt scratch extent used by the pinned BF16 Router arithmetic.
inline constexpr std::size_t kBf16RouterWorkspaceBytes = 1U << 20;

/// Required device-address alignment of the cuBLASLt scratch tensor.
inline constexpr std::size_t kBf16RouterWorkspaceAlignment = 256;

/// Enqueue biased BF16 GEMM using bounded caller-owned device scratch.
/// \pre hidden_states is [M,H], weight is [E,H], bias is [E], and output
/// is [M,E]. All four are nonempty contiguous BF16 tensors on one CUDA device.
/// workspace is a contiguous byte tensor of kBf16RouterWorkspaceBytes bytes
/// on that device, aligned to kBf16RouterWorkspaceAlignment. Output and scratch
/// do not overlap inputs or each other. A successful ordinary launch must warm
/// each device/geometry/alignment combination before capturing that combination.
/// \post output receives the cuBLASLt biased BF16 result on the current CUDA
/// stream. Split-K rounding follows the pinned library's selected algorithm.
/// Device scratch remains owned by the caller; no device allocation or host
/// synchronization occurs. Graph capture fixes the selected launches for replay.
/// \throws c10::Error for invalid tensors, unavailable algorithms, or a library
/// failure, including capture of an unwarmed combination. There is no alternate
/// implementation on failure.
void compute_bf16_router_logits(const at::Tensor &hidden_states, const at::Tensor &weight, const at::Tensor &bias,
                                const at::Tensor &output, const at::Tensor &workspace);

} // namespace xpool::ffnagent
