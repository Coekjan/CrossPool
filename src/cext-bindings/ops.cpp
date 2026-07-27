#include <torch/library.h>

#include <c10/util/TypeCast.h>

#include <cstddef>
#include <cstdint>
#include <optional>

#include <xpool/instance.hpp>
#include <xpool/runtime.hpp>
#include <xpool/transport/protocol.hpp>

namespace {

at::Tensor ffn_shim(const at::Tensor &hidden_states, const std::optional<at::Tensor> &global_num_tokens_gpu,
                    std::int64_t layer_ordinal, std::int64_t forward_mode, std::int64_t result_handoff,
                    std::int64_t dp_padding_mode) {
  xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance, "xpool.ffn_shim");
  const xpool::transport::FfnRequestMetadata request_metadata{
      c10::checked_convert<std::size_t>(layer_ordinal, "layer_ordinal"),
      c10::checked_convert<std::uint32_t>(forward_mode, "forward_mode"),
      c10::checked_convert<std::uint32_t>(result_handoff, "result_handoff"),
      c10::checked_convert<std::uint32_t>(dp_padding_mode, "dp_padding_mode"),
  };
  return xpool::instance::ffn_shim(hidden_states, global_num_tokens_gpu, request_metadata);
}

} // namespace

TORCH_LIBRARY(xpool, module) {
  module.def("ffn_shim(Tensor hidden_states, Tensor? global_num_tokens_gpu, int "
             "layer_ordinal, int forward_mode, int result_handoff, int "
             "dp_padding_mode) -> Tensor");
}

TORCH_LIBRARY_IMPL(xpool, CUDA, module) { module.impl("ffn_shim", TORCH_FN(ffn_shim)); }
