#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <optional>

#include <xpool/debug/loopback.hpp>
#include <xpool/debug/options.hpp>
#include <xpool/instance.hpp>
#include <xpool/transport/instance.hpp>

namespace xpool::instance {

at::Tensor ffn_shim(const at::Tensor &hidden_states, const std::optional<at::Tensor> &global_num_tokens_gpu,
                    const xpool::transport::FfnRequestMetadata &request_metadata) {
  TORCH_CHECK(request_metadata.valid(), "xpool FFN shim received invalid request metadata");
  TORCH_CHECK(hidden_states.is_cuda(), "xpool FFN shim expects a CUDA tensor");
  TORCH_CHECK(hidden_states.dim() == 2, "xpool FFN shim expects a 2D tensor");
  TORCH_CHECK(hidden_states.is_contiguous(), "xpool FFN shim expects a contiguous tensor");
  const auto has_global_num_tokens =
      global_num_tokens_gpu.has_value() && global_num_tokens_gpu->defined() && global_num_tokens_gpu->numel() != 0;
  if (!has_global_num_tokens) {
    TORCH_CHECK(request_metadata.dp_padding_mode == xpool::abi::DpPaddingMode::None,
                "xpool FFN shim DP padding requires global_num_tokens_gpu");
  } else {
    const auto &token_counts = *global_num_tokens_gpu;
    TORCH_CHECK(token_counts.is_cuda(), "xpool FFN shim DP token counts must be a CUDA tensor");
    TORCH_CHECK(token_counts.is_contiguous(), "xpool FFN shim DP token counts must be contiguous");
    TORCH_CHECK(token_counts.dim() == 1, "xpool FFN shim DP token counts must be a 1D tensor");
    TORCH_CHECK(token_counts.scalar_type() == at::kInt || token_counts.scalar_type() == at::kLong,
                "xpool FFN shim DP token counts must be int32 or int64");
    TORCH_CHECK(token_counts.get_device() == hidden_states.get_device(),
                "xpool FFN shim DP token counts must be on the hidden-state device");
  }
  switch (hidden_states.scalar_type()) {
  case c10::ScalarType::BFloat16:
  case c10::ScalarType::Half:
  case c10::ScalarType::Float:
    break;
  default:
    TORCH_CHECK(false, "xpool FFN shim supports only float32, float16, and "
                       "bfloat16 tensors");
  }
  if (xpool::debug::options().loopback.site == xpool::debug::LoopbackSite::Instance) {
    auto output = at::empty_like(hidden_states);
    xpool::debug::launch_loopback_rotation(output, hidden_states);
    return output;
  }
  return xpool::transport::InstanceTransportRuntime::singleton().submit(hidden_states, global_num_tokens_gpu,
                                                                        request_metadata);
}

} // namespace xpool::instance
