#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <optional>

#include <xpool/instance.hpp>
#include <xpool/transport/instance.hpp>

namespace xpool::instance {

void ffn_shim(const at::Tensor &hidden_states, const std::optional<at::Tensor> &dp_rank_payload_rows,
              const at::Tensor &output, const xpool::transport::RequestMetadata &request_metadata) {
  TORCH_CHECK(request_metadata.valid(), "xpool FFN shim received invalid request metadata");
  TORCH_CHECK(hidden_states.is_cuda(), "xpool FFN shim expects a CUDA tensor");
  TORCH_CHECK(hidden_states.dim() == 2, "xpool FFN shim expects a 2D tensor");
  TORCH_CHECK(hidden_states.is_contiguous(), "xpool FFN shim expects a contiguous tensor");
  const auto has_dp_rank_payload_rows =
      dp_rank_payload_rows.has_value() && dp_rank_payload_rows->defined() && dp_rank_payload_rows->numel() != 0;
  if (!has_dp_rank_payload_rows) {
    TORCH_CHECK(request_metadata.dp_row_layout == xpool::ffn::DpRowLayout::None,
                "xpool FFN shim DP row layout requires dp_rank_payload_rows");
  } else {
    const auto &rank_payload_rows = *dp_rank_payload_rows;
    TORCH_CHECK(rank_payload_rows.is_cuda(), "xpool FFN shim DP-rank payload rows must be a CUDA tensor");
    TORCH_CHECK(rank_payload_rows.is_contiguous(), "xpool FFN shim DP-rank payload rows must be contiguous");
    TORCH_CHECK(rank_payload_rows.dim() == 1, "xpool FFN shim DP-rank payload rows must be a 1D tensor");
    TORCH_CHECK(rank_payload_rows.scalar_type() == at::kInt || rank_payload_rows.scalar_type() == at::kLong,
                "xpool FFN shim DP-rank payload rows must be int32 or int64");
    TORCH_CHECK(rank_payload_rows.get_device() == hidden_states.get_device(),
                "xpool FFN shim DP-rank payload rows must be on the hidden-state device");
  }
  switch (hidden_states.scalar_type()) {
  case c10::ScalarType::BFloat16:
  case c10::ScalarType::Half:
    break;
  default:
    TORCH_CHECK(false, "xpool FFN shim supports only float16 and bfloat16 tensors");
  }
  TORCH_CHECK(output.defined(), "xpool FFN shim output must be defined");
  TORCH_CHECK(output.is_cuda(), "xpool FFN shim output must be a CUDA tensor");
  TORCH_CHECK(output.dim() == 2, "xpool FFN shim output must be a 2D tensor");
  TORCH_CHECK(output.is_contiguous(), "xpool FFN shim output must be contiguous");
  TORCH_CHECK(output.sizes() == hidden_states.sizes(), "xpool FFN shim output shape must match hidden states");
  TORCH_CHECK(output.scalar_type() == hidden_states.scalar_type(),
              "xpool FFN shim output dtype must match hidden states");
  TORCH_CHECK(output.device() == hidden_states.device(), "xpool FFN shim output device must match hidden states");
  xpool::transport::InstanceRankRuntime::singleton().submit(hidden_states, dp_rank_payload_rows, output,
                                                                 request_metadata);
}

} // namespace xpool::instance
