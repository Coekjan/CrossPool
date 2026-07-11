#pragma once

/// \file xpool/devagent.hpp
/// \brief Native entrypoints owned by devagent processes.

#include <cstdint>
#include <xpool/transport.hpp>

namespace xpool::devagent {

/// Create one CUDA IPC transport arena owned by the current devagent process.
/// \param cuda_device CUDA device index that owns the arena allocation.
/// \param max_tokens Maximum token rows supported by one slot.
/// \param hidden_size Hidden-state width supported by one slot.
/// \param element_size_bytes Bytes per hidden-state element.
/// \param atn_dp_size Attention data-parallel world size for DP token counts.
/// \return Lowercase hex CUDA IPC arena handle to publish through the daemon.
/// \pre The process is initialized with RuntimeRole::kDevagent and all geometry
/// values are positive.
/// \post The returned handle identifies CUDA storage owned by this process
/// until destroy_transport_arena completes. \throws c10::Error on invalid
/// geometry, duplicate state, or CUDA failure.
xpool::transport::TransportArenaHandleHex create_transport_arena(
    std::int64_t cuda_device, std::int64_t max_tokens, std::int64_t hidden_size,
    std::int64_t element_size_bytes, std::int64_t atn_dp_size);

/// Destroy one CUDA IPC transport arena owned by this devagent process.
/// \param handle Lowercase hex CUDA IPC handle identifying the arena.
/// \return Structured final trace snapshot; records are empty when observation
/// was disabled.
/// \post The resident kernel is drained, observer state is copied to CPU when
/// enabled, and the handle no longer identifies live CUDA storage.
/// \throws c10::Error when the handle is unknown or already destroyed, or when
/// kernel drain, snapshot copy, or CUDA release fails.
xpool::abi::TransportTraceSnapshot destroy_transport_arena(
    const xpool::transport::TransportArenaHandleHex &handle);

/// Launch the persistent transport kernel for one devagent-owned arena.
/// \param handle Lowercase hex CUDA IPC handle identifying the arena.
/// \pre create_transport_arena returned handle and no kernel is running for it.
/// \post One resident kernel remains active until arena destruction.
/// \throws c10::Error when the handle is unknown or CUDA launch fails.
void launch_transport_kernel(
    const xpool::transport::TransportArenaHandleHex &handle);

} // namespace xpool::devagent
