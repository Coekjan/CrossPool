#include <cstddef>
#include <cstdint>

#include <cooperative_groups.h>

#include <xpool/abi.hpp>
#include <xpool/debug/loopback.cuh>
#include <xpool/debug/options.cuh>
#include <xpool/ffnagent/executor.cuh>
#include <xpool/macros.hpp>

namespace xpool::ffnagent::executor {

XPOOL_DEVICE_FN xpool::abi::FfnResultCode execute(
    const cooperative_groups::thread_block &group,
    const xpool::fabric::FfnInvocation &invocation,
    const xpool::fabric::FabricModelLayout &model,
    const xpool::fabric::FabricLayerLayout &layer,
    const void *input,
    void *output) {
  const auto validation_result = invocation.validate();
  if (validation_result != xpool::abi::FfnResultCode::Ok) {
    return validation_result;
  }
  if (!xpool::fabric::FfnLayerKind::is_valid(layer.kind)) {
    return xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch};
  }
  if (xpool::debug::options().loopback.site != xpool::debug::LoopbackSite::FfnAgent) {
    return xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::NotImplemented};
  }
  if (model.hidden_size % 2 != 0) {
    return xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::NotImplemented};
  }
  const auto pair_count = (invocation.payload_rows * model.hidden_size) / std::size_t{2};
  xpool::debug::rotate_hidden_pairs(
      group, output, input, xpool::abi::TensorDType{model.dtype}, pair_count);
  return xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok};
}

} // namespace xpool::ffnagent::executor
