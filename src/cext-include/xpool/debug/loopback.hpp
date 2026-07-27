#pragma once

/// \file xpool/debug/loopback.hpp
/// \brief Test-oriented FFN loopback executor used to validate shim dispatch.
///
/// The loopback executor is a development validation surface only. It proves
/// tensor dispatch, CUDA graph replay, and native op registration before xpool
/// wires the real agent FFN executor.

#include <ATen/core/TensorBody.h>

namespace xpool::debug {

/// Launch the pairwise 45-degree hidden-state rotation CUDA kernel.
///
/// \param output Contiguous CUDA output tensor with the same shape and dtype as
/// input.
/// \param input Contiguous CUDA input tensor with shape
/// `[num_tokens, hidden_size]`; hidden_size must be even.
/// \pre output and input are CUDA tensors on the same device.
/// \pre output and input are contiguous and have identical shape and dtype.
/// \pre input and output dtype is float32, float16, or bfloat16.
/// \pre input has rank 2 and an even hidden dimension.
/// \post output contains pairwise `(x - y) / sqrt(2), (x + y) / sqrt(2)`
/// values. \throws c10::Error if CUDA launch configuration or dispatch fails.
/// \remark Side effect: enqueues work on the current CUDA stream for input's
/// device.
void launch_loopback_rotation(const at::Tensor &output, const at::Tensor &input);

} // namespace xpool::debug
