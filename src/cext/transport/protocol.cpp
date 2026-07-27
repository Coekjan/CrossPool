#include <cstddef>

#include <c10/util/Exception.h>

#include <xpool/transport/protocol.hpp>

namespace xpool::transport {

void FfnRequestMetadata::validate(const TransportArenaLayout &layout, std::size_t payload_rows,
                                  bool token_counts_present) const {
  TORCH_CHECK(valid(), "xpool FFN request contains an invalid mode or result handoff");
  TORCH_CHECK(payload_rows != 0 && payload_rows <= layout.max_tokens,
              "xpool FFN request row count exceeds Transport capacity");
  if (layout.atn_dp_size == 1) {
    TORCH_CHECK(dp_padding_mode == xpool::abi::DpPaddingMode::None && !token_counts_present,
                "xpool DP-one request must not carry padding or token counts");
    return;
  }
  TORCH_CHECK(dp_padding_mode == xpool::abi::DpPaddingMode::MaxLen ||
                  dp_padding_mode == xpool::abi::DpPaddingMode::SumLen,
              "xpool DP request requires MAX_LEN or SUM_LEN padding");
  TORCH_CHECK(token_counts_present, "xpool DP request requires one token count per attention DP rank");
}

} // namespace xpool::transport
