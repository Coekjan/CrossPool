#pragma once

/// \file xpool/instance.hpp
/// \brief Native entrypoints owned by instance processes.

#include <ATen/core/TensorBody.h>

#include <cstdint>
#include <optional>

#include <xpool/transport/protocol.hpp>
namespace xpool::instance {

/// Execute one FFN shim request by dispatching on the hidden-state device.
/// \param hidden_states Hidden-state tensor shaped [tokens, hidden_size].
/// \param global_num_tokens_gpu Optional CUDA DP token-counts tensor.
/// \param request_metadata Request metadata stamped into the device-visible
/// request.
/// \return Output tensor completed by the selected FFN shim runtime path.
/// \pre hidden_states is a contiguous rank-2 CUDA tensor whose metadata fits
/// the attached arena; global_num_tokens_gpu is contiguous CUDA int32 or int64 when
/// present. \post Native work is ordered on the current CUDA stream without
/// host sync. \throws c10::Error when no arena is attached or metadata violates
/// the ABI.
at::Tensor ffn_shim(const at::Tensor &hidden_states, const std::optional<at::Tensor> &global_num_tokens_gpu,
                    const xpool::transport::FfnRequestMetadata &request_metadata);

} // namespace xpool::instance
