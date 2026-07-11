#include <ATen/ATen.h>

#include <cstdint>

#include <xpool/abi.hpp>
#include <xpool/debug/loopback.cuh>
#include <xpool/debug/options.cuh>
#include <xpool/transport/executor.cuh>
#include <xpool/transport/protocol.cuh>
#include <xpool/utils/device.cuh>

namespace xpool::transport {

namespace {

template <typename scalar_t>
__device__ void
execute_loopback_request(const TransportArena &arena,
                         const xpool::abi::FfnRequestDescriptor &request) {
  const auto *input = arena.request_input<scalar_t>(request);
  auto *output = arena.request_output<scalar_t>(request);
  const std::int64_t pair_count =
      (static_cast<std::int64_t>(request.tensor_metadata.num_tokens) *
       static_cast<std::int64_t>(request.tensor_metadata.hidden_size)) /
      2;
  xpool::debug::rotate_hidden_pairs_strided(
      output, input, static_cast<std::int64_t>(threadIdx.x),
      static_cast<std::int64_t>(blockDim.x), pair_count);
}

} // namespace

__device__ std::uint32_t
execute_transport_request(const TransportArena &arena,
                          const xpool::abi::FfnRequestDescriptor &request) {
  if (xpool::debug::g_debug_options.enabled(
          xpool::abi::DebugOption::kTransportLoopback)) {
    switch (request.tensor_metadata.dtype.value) {
    case xpool::abi::TensorDType::kFp32:
      execute_loopback_request<float>(arena, request);
      break;
    case xpool::abi::TensorDType::kFp16:
      execute_loopback_request<at::Half>(arena, request);
      break;
    case xpool::abi::TensorDType::kBf16:
      execute_loopback_request<at::BFloat16>(arena, request);
      break;
    default:
      xpool::utils::device::trap();
    }
    return xpool::abi::FfnResultErrorCode::kOk;
  }
  return xpool::abi::FfnResultErrorCode::kNotImplemented;
}

} // namespace xpool::transport
