#pragma once

/// \file xpool/instance.hpp
/// \brief Native entrypoints owned by instance processes.

#include <ATen/core/TensorBody.h>

#include <cstdint>
#include <optional>

#include <xpool/abi.hpp>
#include <xpool/transport.hpp>

namespace xpool::instance {

/// Attach one daemon-brokered transport arena to an instance rank.
/// Reattaching an identical arena is idempotent. Attaching a different arena
/// for an existing instance-rank key fails; callers must explicitly detach the
/// old arena before a restarted instance can attach a new one.
/// \param instance_index Integer instance index from resolved xpool config.
/// \param rank Rank-local process index within the instance.
/// \param handle Lowercase hex CUDA IPC arena handle acquired from the daemon.
/// \post The key owns an open CUDA IPC mapping for handle.
/// \throws c10::Error when the handle is invalid or the key owns another arena.
void attach_transport_arena(
    std::int64_t instance_index, std::int64_t rank,
    const xpool::transport::TransportArenaHandleHex &handle);

/// Detach one daemon-brokered transport arena from an instance rank.
/// \param instance_index Integer instance index from resolved xpool config.
/// \param rank Rank-local process index within the instance.
/// \post New launches are stopped, in-flight work is complete, and the CUDA IPC
/// mapping is closed. Missing keys are idempotent.
/// \throws c10::Error when synchronization or CUDA IPC cleanup fails.
void detach_transport_arena(std::int64_t instance_index, std::int64_t rank);

/// Return the sticky transport executor error for one attached arena.
/// \param instance_index Integer instance index from resolved xpool config.
/// \param rank Rank-local process index within the instance.
/// \return FfnResultErrorCode value, or kOk when no error was recorded.
std::uint32_t transport_error_snapshot(std::int64_t instance_index,
                                       std::int64_t rank);

/// Execute one FFN shim request by dispatching on the hidden-state device.
/// \param hidden_states Hidden-state tensor shaped [tokens, hidden_size].
/// \param request_metadata Request metadata stamped into the device-visible
/// descriptor.
/// \param rank Rank-local process index within the instance.
/// \param global_num_tokens_gpu Optional CUDA DP token-counts tensor.
/// \return Output tensor completed by the selected FFN shim runtime path.
/// \pre hidden_states is a contiguous rank-2 CUDA tensor whose metadata fits
/// the attached arena; global_num_tokens_gpu is contiguous CUDA int32 when
/// present. \post Native work is ordered on the current CUDA stream without
/// host sync. \throws c10::Error when no arena is attached or metadata violates
/// the ABI.
at::Tensor ffn_shim(const at::Tensor &hidden_states,
                    const xpool::abi::FfnRequestMetadata &request_metadata,
                    std::int64_t rank,
                    const std::optional<at::Tensor> &global_num_tokens_gpu);

} // namespace xpool::instance
