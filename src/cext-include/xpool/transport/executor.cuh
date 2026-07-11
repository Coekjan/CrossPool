#pragma once

/// \file xpool/transport/executor.cuh
/// \brief Device-side transport executor boundary.

#include <cstdint>

#include <xpool/abi.hpp>
#include <xpool/transport/protocol.cuh>

namespace xpool::transport {

/// Execute one granted transport request on the devagent resident kernel.
/// \param arena Device view of the shared transport arena.
/// \param request Granted request descriptor whose payload offsets are valid.
/// \return FfnResultErrorCode value written to the result descriptor.
__device__ std::uint32_t
execute_transport_request(const TransportArena &arena,
                          const xpool::abi::FfnRequestDescriptor &request);

} // namespace xpool::transport
