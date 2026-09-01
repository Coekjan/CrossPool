#pragma once

/// \file xpool/instance.hpp
/// \brief Native entrypoints owned by instance processes.

#include <ATen/core/TensorBody.h>

#include <cstdint>
#include <optional>

#include <xpool/transport/protocol.hpp>
namespace xpool::instance {

/// Execute one FFN shim request by dispatching on the hidden-state device.
/// \pre hidden_states is a contiguous rank-2 CUDA tensor whose metadata fits
/// the attached arena; dp_rank_payload_rows is contiguous CUDA int32 or int64
/// when present; output matches hidden_states in shape, dtype, device, and
/// contiguity.
/// \post Native work is ordered on the current CUDA stream without host sync.
/// \throws c10::Error when no arena is attached or metadata violates the ABI.
void ffn_shim(const at::Tensor &hidden_states, const std::optional<at::Tensor> &dp_rank_payload_rows,
              const at::Tensor &output, const xpool::transport::RequestMetadata &request_metadata);

} // namespace xpool::instance
