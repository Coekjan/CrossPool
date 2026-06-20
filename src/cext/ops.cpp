/// \file ops.cpp
/// \brief Torch operator registration for xpool native FFN shim ops.

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cstdint>

#include <abi.hpp>
#include <loopback.hpp>

namespace {

/// Return the native ABI version used by Python startup preflight.
std::int64_t abi_version() {
  return static_cast<std::int64_t>(xpool::kAbiVersion);
}

bool is_supported_forward_mode(std::int64_t forward_mode) {
  return forward_mode ==
             static_cast<std::int64_t>(xpool::ForwardMode::kDecode) ||
         forward_mode == static_cast<std::int64_t>(xpool::ForwardMode::kExtend);
}

bool is_supported_hidden_dtype(at::ScalarType scalar_type) {
  return scalar_type == at::kFloat || scalar_type == at::kHalf ||
         scalar_type == at::kBFloat16;
}

[[noreturn]] at::Tensor ffn_shim_unimplemented(const at::Tensor &hidden_states,
                                               std::int64_t instance_index,
                                               std::int64_t model_index,
                                               std::int64_t layer_id,
                                               std::int64_t forward_mode) {
  (void)hidden_states;
  (void)instance_index;
  (void)model_index;
  (void)layer_id;
  (void)forward_mode;
  TORCH_CHECK(
      false,
      "xpool ffn_shim is not implemented yet; set "
      "XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE=1 to validate the debug loopback op");
}

void validate_loopback_ffn_contract(const at::Tensor &hidden_states,
                                    std::int64_t forward_mode) {
  TORCH_CHECK(
      is_supported_forward_mode(forward_mode),
      "xpool loopback FFN shim only supports DECODE and EXTEND forward modes");
  TORCH_CHECK(hidden_states.dim() == 2,
              "xpool loopback FFN shim expects a 2D tensor");
  TORCH_CHECK(hidden_states.is_contiguous(),
              "xpool loopback FFN shim expects a contiguous tensor");
  TORCH_CHECK(is_supported_hidden_dtype(hidden_states.scalar_type()),
              "xpool loopback FFN shim supports only float32, float16, and "
              "bfloat16 tensors");
  TORCH_CHECK(hidden_states.size(1) % 2 == 0,
              "xpool loopback FFN shim 45-degree rotation requires an even "
              "hidden size");
}

at::Tensor ffn_shim_loopback_cuda(const at::Tensor &hidden_states,
                                  std::int64_t instance_index,
                                  std::int64_t model_index,
                                  std::int64_t layer_id,
                                  std::int64_t forward_mode) {
  (void)instance_index;
  (void)model_index;
  (void)layer_id;
  validate_loopback_ffn_contract(hidden_states, forward_mode);
  TORCH_CHECK(hidden_states.is_cuda(),
              "xpool loopback FFN shim expects a CUDA tensor");

  at::Tensor output = at::empty_like(hidden_states);
  xpool::launch_loopback_rotation(output, hidden_states);
  return output;
}

at::Tensor ffn_shim_loopback_meta(const at::Tensor &hidden_states,
                                  std::int64_t instance_index,
                                  std::int64_t model_index,
                                  std::int64_t layer_id,
                                  std::int64_t forward_mode) {
  (void)instance_index;
  (void)model_index;
  (void)layer_id;
  validate_loopback_ffn_contract(hidden_states, forward_mode);
  return at::empty_like(hidden_states);
}

} // namespace

/// Register the xpool Torch operator schemas.
TORCH_LIBRARY(xpool, m) {
  m.def("abi_version() -> int");
  m.impl("abi_version", abi_version);
  m.def("ffn_shim(Tensor hidden_states, int instance_index, int model_index, "
        "int layer_id, int forward_mode) -> Tensor");
  m.def("ffn_shim_loopback(Tensor hidden_states, int instance_index, int "
        "model_index, int layer_id, int forward_mode) -> Tensor");
}

/// Register CUDA implementations for xpool Torch operators.
TORCH_LIBRARY_IMPL(xpool, CUDA, m) {
  m.impl("ffn_shim", ffn_shim_unimplemented);
  m.impl("ffn_shim_loopback", ffn_shim_loopback_cuda);
}

/// Register Meta implementations for xpool Torch operators.
TORCH_LIBRARY_IMPL(xpool, Meta, m) {
  m.impl("ffn_shim", ffn_shim_unimplemented);
  m.impl("ffn_shim_loopback", ffn_shim_loopback_meta);
}
