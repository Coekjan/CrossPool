#include <cstddef>
#include <cstdint>
#include <optional>

#include <c10/util/TypeCast.h>
#include <torch/library.h>

#include <xpool/instance.hpp>
#include <xpool/runtime.hpp>
#include <xpool/transport/protocol.hpp>

namespace {

void ffn_shim(const at::Tensor &hidden_states, const std::optional<at::Tensor> &dp_rank_payload_rows,
              const at::Tensor &output, std::int64_t layer_ordinal, std::int64_t forward_mode,
              std::int64_t output_requirement, std::int64_t dp_row_layout) {
  xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance, "xpool.ffn_shim");
  const auto parsed_forward_mode =
      static_cast<xpool::ffn::ForwardMode>(c10::checked_convert<std::uint32_t>(forward_mode, "forward_mode"));
  const auto parsed_output_requirement = static_cast<xpool::ffn::OutputRequirement>(
      c10::checked_convert<std::uint32_t>(output_requirement, "output_requirement"));
  const auto parsed_dp_row_layout =
      static_cast<xpool::ffn::DpRowLayout>(c10::checked_convert<std::uint32_t>(dp_row_layout, "dp_row_layout"));
  TORCH_CHECK(xpool::ffn::is_valid(parsed_forward_mode), "xpool ffn_shim received an invalid forward mode");
  TORCH_CHECK(xpool::ffn::is_valid(parsed_output_requirement), "xpool ffn_shim received an invalid output requirement");
  TORCH_CHECK(xpool::ffn::is_valid(parsed_dp_row_layout), "xpool ffn_shim received an invalid DP row layout");
  const xpool::transport::RequestMetadata request_metadata{
      c10::checked_convert<std::size_t>(layer_ordinal, "layer_ordinal"),
      parsed_forward_mode,
      parsed_output_requirement,
      parsed_dp_row_layout,
  };
  xpool::instance::ffn_shim(hidden_states, dp_rank_payload_rows, output, request_metadata);
}

} // namespace

TORCH_LIBRARY(xpool, module) {
  module.def("ffn_shim(Tensor hidden_states, Tensor? dp_rank_payload_rows, Tensor(a!) output, int "
             "layer_ordinal, int forward_mode, int output_requirement, int "
             "dp_row_layout) -> ()");
}

TORCH_LIBRARY_IMPL(xpool, CUDA, module) { module.impl("ffn_shim", TORCH_FN(ffn_shim)); }
