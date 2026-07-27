#include <array>
#include <cstddef>
#include <cstdint>

#include <c10/util/Exception.h>

#include <xpool/debug/options.hpp>
#include <xpool/transport/arena.hpp>
#include <xpool/transport/layout.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/layout.hpp>

namespace xpool::transport {

namespace {

struct TransportArenaRegions {
  xpool::utils::layout::LayoutRegion layout;
  xpool::utils::layout::LayoutRegion state;
  xpool::utils::layout::LayoutRegion mailbox;
  xpool::utils::layout::LayoutRegion input_payload;
  xpool::utils::layout::LayoutRegion output_payload;
  xpool::utils::layout::LayoutRegion dp_token_counts;
  xpool::utils::layout::LayoutRegion trace_records;
  std::size_t total_bytes;

  TransportArenaRegions(std::size_t payload_bytes, std::size_t dp_token_count,
                        std::size_t trace_capacity) {
    using xpool::utils::layout::LayoutRegionSpec;
    const auto specs = std::to_array<LayoutRegionSpec>({
        LayoutRegionSpec::object<TransportArenaLayout>("transport layout"),
        LayoutRegionSpec::object<TransportArenaState>("transport state"),
        LayoutRegionSpec::object<TransportMailbox>("transport mailbox"),
        LayoutRegionSpec::bytes("transport input payload", payload_bytes,
                                xpool::arena::kPayloadAlignment),
        LayoutRegionSpec::bytes("transport output payload", payload_bytes,
                                xpool::arena::kPayloadAlignment),
        LayoutRegionSpec::array<std::uint32_t>("transport DP token counts", dp_token_count),
        LayoutRegionSpec::array<TransportTraceRecord>("transport trace records", trace_capacity),
    });
    const xpool::utils::layout::LayoutPlan plan{
        specs, xpool::arena::kAllocationAlignment};
    auto index = std::size_t{0};
    layout = plan[index++];
    state = plan[index++];
    mailbox = plan[index++];
    input_payload = plan[index++];
    output_payload = plan[index++];
    dp_token_counts = plan[index++];
    trace_records = plan[index++];
    total_bytes = plan.total_bytes;
    TORCH_CHECK(index == plan.regions.size(), "xpool transport regions did not consume every entry");
  }
};

std::size_t payload_bytes(std::size_t max_tokens, std::size_t hidden_size,
                          xpool::abi::TensorDType dtype) {
  return xpool::utils::checked::prod(max_tokens, hidden_size, dtype.bytes());
}

} // namespace

TransportArenaLayout TransportArenaLayout::create(
    std::size_t instance_index, std::size_t instance_rank, std::size_t atn_tp_rank,
    std::size_t atn_tp_size, std::size_t atn_dp_rank, std::size_t atn_dp_size,
    std::size_t max_tokens, std::size_t hidden_size, xpool::abi::TensorDType dtype) {
  TORCH_CHECK(max_tokens != 0, "xpool transport arena requires a positive token capacity");
  TORCH_CHECK(hidden_size != 0, "xpool transport arena requires a positive hidden size");
  TORCH_CHECK(atn_tp_size != 0 && atn_tp_rank < atn_tp_size,
              "xpool transport arena has invalid attention TP topology");
  TORCH_CHECK(atn_dp_size != 0 && atn_dp_rank < atn_dp_size,
              "xpool transport arena has invalid attention DP topology");

  const auto bytes = payload_bytes(max_tokens, hidden_size, dtype);
  const auto dp_token_count = atn_dp_size == 1 ? std::size_t{0} : atn_dp_size;
  const auto trace_capacity = xpool::debug::options().transport_observer.capacity();
  const TransportArenaRegions regions{bytes, dp_token_count, trace_capacity};
  return TransportArenaLayout{
      .header = {.magic = kTransportArenaMagic,
                 .abi_version = xpool::abi::kAbiVersion,
                 .layout_size = sizeof(TransportArenaLayout),
                 .total_bytes = regions.total_bytes,
                 .state_offset = regions.state.offset},
      .instance_index = instance_index,
      .instance_rank = instance_rank,
      .atn_tp_rank = atn_tp_rank,
      .atn_tp_size = atn_tp_size,
      .atn_dp_rank = atn_dp_rank,
      .atn_dp_size = atn_dp_size,
      .max_tokens = max_tokens,
      .hidden_size = hidden_size,
      .dtype = dtype.value(),
      .mailbox_offset = regions.mailbox.offset,
      .input_payload_offset = regions.input_payload.offset,
      .output_payload_offset = regions.output_payload.offset,
      .dp_token_counts_offset = dp_token_count == 0 ? 0 : regions.dp_token_counts.offset,
      .trace = {.records_offset = trace_capacity == 0 ? 0 : regions.trace_records.offset,
                .capacity = trace_capacity},
  };
}

void TransportArenaLayout::validate() const {
  TORCH_CHECK(header.magic == kTransportArenaMagic, "xpool transport arena header magic does not match");
  TORCH_CHECK(header.abi_version == xpool::abi::kAbiVersion &&
                  header.layout_size == sizeof(TransportArenaLayout),
              "xpool transport arena layout has an incompatible ABI");
  TORCH_CHECK(max_tokens != 0 && hidden_size != 0 && xpool::abi::TensorDType::is_valid(dtype),
              "xpool transport arena layout has invalid tensor geometry");
  TORCH_CHECK(atn_tp_size != 0 && atn_tp_rank < atn_tp_size,
              "xpool transport arena layout has invalid attention TP topology");
  TORCH_CHECK(atn_dp_size != 0 && atn_dp_rank < atn_dp_size,
              "xpool transport arena layout has invalid attention DP topology");

  const auto bytes = payload_bytes(max_tokens, hidden_size, xpool::abi::TensorDType{dtype});
  const auto dp_token_count = atn_dp_size == 1 ? std::size_t{0} : atn_dp_size;
  const TransportArenaRegions regions{bytes, dp_token_count, trace.capacity};
  const auto expected_dp_offset = dp_token_count == 0 ? std::size_t{0} : regions.dp_token_counts.offset;
  const auto expected_trace_offset = trace.capacity == 0 ? std::size_t{0} : regions.trace_records.offset;
  TORCH_CHECK(header.total_bytes == regions.total_bytes && header.state_offset == regions.state.offset &&
                  mailbox_offset == regions.mailbox.offset &&
                  input_payload_offset == regions.input_payload.offset &&
                  output_payload_offset == regions.output_payload.offset &&
                  dp_token_counts_offset == expected_dp_offset &&
                  trace.records_offset == expected_trace_offset,
              "xpool transport arena layout does not match canonical geometry");
}

} // namespace xpool::transport
